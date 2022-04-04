#include <torch/csrc/lazy/ts_backend/ops/device_data.h>

#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

#include <sstream>

namespace torch {
namespace lazy {

DeviceData::DeviceData(std::shared_ptr<BackendData> data)
    : TsNode(
          ltc_device_data,
          data->shape(),
          /*num_outputs=*/1,
          /*hash_seed=*/static_cast<uint32_t>(101)),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", device=" << data_->device();
  return ss.str();
}

const DeviceData* DeviceData::Cast(const Node* node) {
  return NodeCast<DeviceData>(node, ltc_device_data);
}

NodePtr DeviceData::Create(std::shared_ptr<BackendData> data) {
  NodePtr node = nullptr;
  if (FLAGS_torch_lazy_reuse_ir) {
    node = ReuseNode<DeviceData>(*ltc_device_data, data);
  }
  if (!node) {
    node = MakeNode<DeviceData>(std::move(data));
  }
  return node;
}

} // namespace lazy
} // namespace torch
