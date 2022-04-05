# Owner(s): ["oncall: distributed"]

import sys
from contextlib import suppress
from copy import deepcopy
from functools import partial
from typing import Any, Dict

import torch
from torch import distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    CPUOffload,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.nn import Linear, Module
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_full_params,
    _get_full_detached_param,
    _get_state_dict,
    SkipModel,
    _zero_model,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

INNER_SHAPE = [4, 4]
OUTER_SHAPE = [4, 5]

_SUPPORTED_STATE_DICT_IMPLS = ["state_dict", "local_state_dict"]

STATE_DICT_MAPPING = {
    "state_dict": StateDictType.FULL_STATE_DICT,
    "local_state_dict": StateDictType.LOCAL_STATE_DICT,
    "sharded_state_dict": StateDictType.SHARDED_STATE_DICT,
}


class Model(Module):
    def __init__(self, wrap_fsdp):
        super().__init__()
        self.inner = Linear(*INNER_SHAPE)
        if wrap_fsdp:
            self.inner = FSDP(self.inner)
        self.outer = Linear(*OUTER_SHAPE)

    def forward(self, x):
        # Forward twice.
        i = self.inner(x)
        j = self.inner(x)
        return self.outer(i + j)


class TestFSDPStateDict(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _get_simple_nested_model(self, *fsdp_args, **fsdp_kwargs):
        model = FSDP(
            nn.Sequential(
                FSDP(nn.Linear(10, 10, bias=False).cuda(), *fsdp_args, **fsdp_kwargs),
                nn.Linear(10, 10, bias=False).cuda(),
            ),
            *fsdp_args,
            **fsdp_kwargs,
        )
        return model

    def _get_simple_model(self, *fsdp_args, **fsdp_kwargs):
        model = FSDP(nn.Linear(10, 10, bias=False).cuda(), *fsdp_args, **fsdp_kwargs)
        return model

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("fp16", [True, False])
    def test_basic_save_and_load_state_dict(self, cpu_offload, fp16):
        """
        Tests that we can save a state_dict and load it into a blank model
        with various configs such as fp16 and cpu offload and parameters
        match as expected.
        """
        for model_call in [
            partial(self._get_simple_nested_model, cpu_offload=cpu_offload),
            partial(self._get_simple_model, cpu_offload=cpu_offload),
        ]:
            model = model_call()
            fsdp_state_dict = _get_state_dict(model, cpu_offload.offload_params, fp16)
            if fp16:
                # Verify fp16 is the type
                for tensor in fsdp_state_dict.values():
                    self.assertEqual(tensor.dtype, torch.float16)

            model_new = model_call()
            if not cpu_offload.offload_params:
                model_new = model_new.cuda()
            if fp16:
                model_new.half()

            # zero the model to ensure parameters are different.
            _zero_model(model_new)

            with model.summon_full_params(), model_new.summon_full_params():
                params = list(model.parameters())
                params_new = list(model_new.parameters())
                self.assertNotEqual(params, params_new)

            # Verify parameters are the same in the new model.
            model_new.load_state_dict(fsdp_state_dict)
            with model_new.summon_full_params():
                with model.summon_full_params():
                    params = list(model.parameters())
                    params_new = list(model_new.parameters())
                    self.assertEqual(params, params_new)
                    if fp16:
                        for tensor in model_new.parameters():
                            self.assertEqual(tensor.dtype, torch.float16)

    @skip_if_lt_x_gpu(2)
    @parametrize("mixed_precision", [True, False])
    def test_save_and_load_after_forward_state_dict(self, mixed_precision):
        """
        Test that saving after some training results in params being updated as
        expected.
        """
        torch.cuda.set_device(self.rank)
        mixed_precision = MixedPrecision() if mixed_precision else None
        model = self._get_simple_nested_model(mixed_precision=mixed_precision)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_params = _get_full_detached_param(model)
        for _ in range(6):
            inp = torch.randn(1, 10, device=torch.cuda.current_device())
            output = model(*inp)
            loss = output.sum()
            expected_dtype = torch.float32 if mixed_precision is None else torch.float16
            self.assertEqual(expected_dtype, loss.dtype)
            loss.backward()
            optim.step()

        trained_params = _get_full_detached_param(model)
        # Ensure some training occured
        self.assertNotEqual(initial_params, trained_params)
        # Save a copy of the state_dict
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        _zero_model(model)

        # Ensure checkpointed params have the full param dtype
        for tensor in state_dict.values():
            self.assertEqual(tensor.dtype, torch.float32)

        # Load state_dict into zeroed model
        model.load_state_dict(state_dict)
        loaded_params = _get_full_detached_param(model)
        self.assertEqual(loaded_params, trained_params)

    def _initialize_model(self, wrap_fsdp: bool, wrap_ddp: bool = True):
        # keep everything deterministic for input data
        torch.manual_seed(0)

        model = Model(wrap_fsdp).cuda()
        if wrap_fsdp:
            model = FSDP(model)
        elif wrap_ddp:
            model = DistributedDataParallel(model, device_ids=[self.rank])
        return model

    @staticmethod
    def _state_dict(model: Module, state_dict_type: str):
        try:
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError:
            raise ValueError(f"No state_dict type for {state_dict_type}")

        with FSDP.state_dict_type(model, enum_val):
            return model.state_dict()

    @staticmethod
    def _load_state_dict(
        model: Module, state_dict_type: str, state_dict: Dict[str, Any]
    ):
        try:
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError:
            raise ValueError(f"No state_dict for {state_dict_type}")

        with FSDP.state_dict_type(model, enum_val):
            return model.load_state_dict(state_dict)

    def _dist_train(self, wrap_fsdp: bool, state_dict_type: str = ""):
        # TODO: Move this test to common_fsdp.
        model = self._initialize_model(wrap_fsdp)
        optim = SGD(model.parameters(), lr=0.1)

        in_data = torch.rand(64, 4, requires_grad=True, device=torch.device("cuda"))
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

        if wrap_fsdp:
            blank_model = FSDP(Model(True).cuda())
            _zero_model(blank_model)
            state_dict = self._state_dict(model, state_dict_type)
            self._load_state_dict(blank_model, state_dict_type, state_dict)
            return get_full_params(blank_model)
        else:
            return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    def test_state_dict_save_load_flow(self, state_dict_type):
        fsdp_params = self._dist_train(wrap_fsdp=True, state_dict_type=state_dict_type)
        ddp_params = self._dist_train(wrap_fsdp=False)
        self.assertEqual(ddp_params, fsdp_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    def test_fsdp_state_dict_keys(self, state_dict_type):
        state_dict = self._state_dict(self._initialize_model(True), state_dict_type)
        if state_dict_type == "local_state_dict":
            self.assertEqual(set(["flat_param", "inner.flat_param"]), state_dict.keys())
        elif state_dict_type == "state_dict":
            # Keys should match local model.
            local_model = self._initialize_model(wrap_fsdp=False, wrap_ddp=False)
            local_keys = local_model.state_dict().keys()
            self.assertEqual(state_dict.keys(), local_keys)
        else:
            raise NotImplementedError(f"No test for {state_dict_type}!")

    @skip_if_lt_x_gpu(2)
    def test_state_dict_load_into_local_module(self):
        """
        Tests that FSDP's state_dict can be loaded into a local model.
        """
        model = self._initialize_model(wrap_fsdp=True)
        optim = SGD(model.parameters(), lr=0.1)
        in_data = torch.rand(64, 4, requires_grad=True, device=torch.device("cuda"))
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

        with model.summon_full_params():
            fsdp_params = deepcopy(list(model.parameters()))

        # get FSDP state_dict. Note that by default we return state_dict.
        fsdp_state_dict = model.state_dict()
        # Create zeroed local model
        blank_local_model = self._initialize_model(wrap_fsdp=False, wrap_ddp=False)
        for param in blank_local_model.parameters():
            with torch.no_grad():
                param.zero_()

        # Load fsdp's full state dict into the local and verify params are as
        # expected.
        blank_local_model.load_state_dict(fsdp_state_dict)
        local_params = list(blank_local_model.parameters())
        for fsdp_param, local_param in zip(fsdp_params, local_params):
            self.assertEqual(fsdp_param, local_param)

    @skip_if_lt_x_gpu(2)
    @parametrize("double_nest", [True])
    def test_state_dict_skip_module(self, double_nest):
        torch.cuda.set_device(self.rank)

        def _create_module(wrap_fsdp=True):
            LINEAR_SKIP = "linear_skip"
            ctx = enable_wrap(wrapper_cls=FSDP) if wrap_fsdp else suppress()
            with ctx:
                module = SkipModel(double_nest=double_nest)
                # Full name of linear_skip param tensors in SkipModel, as would be
                # stored in checkpoint.
                linear_skip_tensor_names = [
                    k for k in dict(module.named_parameters()).keys()
                    if LINEAR_SKIP in k
                ]
                # skip SkipModule
                linear_skip = getattr(module, LINEAR_SKIP)
                delattr(module, LINEAR_SKIP)
                # Wrap FSDP
                fsdp = wrap(module)
                # reattach
                setattr(module, LINEAR_SKIP, linear_skip)
                return fsdp, linear_skip_tensor_names

        fsdp, linear_skip_tensor_names = _create_module()
        # Run a forward pass
        inp = torch.randn((1, 10), device=torch.cuda.current_device())
        loss = fsdp(inp)
        loss.sum().backward()

        state_dict = fsdp.state_dict()
        if self.rank == 0:
            sd_keys = list(state_dict.keys())
            expected = list(SkipModel(double_nest=False).state_dict().keys())
            self.assertEqual(sorted(sd_keys), sorted(expected))
            # TODO: parameters in linear_skip_tensor_names should not be handled
            # by FSDP.state_dict(). Have a check once this is implemented in
            # FSDP.state_dict().

        # Check that it can be loaded into FSDP.
        new_fsdp, _ = _create_module()
        _zero_model(new_fsdp)
        for (p1, p2) in zip(fsdp.parameters(), new_fsdp.parameters()):
            self.assertNotEqual(p1, p2)
        new_fsdp.load_state_dict(deepcopy(state_dict))
        for (p1, p2) in zip(fsdp.parameters(), new_fsdp.parameters()):
            self.assertEqual(p1, p2)

        # Test that the checkpoint can be loaded into a local model.
        local, _ = _create_module(wrap_fsdp=False)
        for param in local.parameters():
            with torch.no_grad():
                param.zero_()

        with fsdp.summon_full_params():
            for (p1, p2) in zip(fsdp.parameters(), local.parameters()):
                self.assertNotEqual(p1, p2)

        local.load_state_dict(deepcopy(state_dict))
        with fsdp.summon_full_params():
            for (p1, p2) in zip(fsdp.parameters(), local.parameters()):
                self.assertEqual(p1, p2)


instantiate_parametrized_tests(TestFSDPStateDict)

if __name__ == "__main__":
    run_tests()
