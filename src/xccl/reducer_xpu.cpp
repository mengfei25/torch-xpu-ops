#include <torch/csrc/distributed/c10d/reducer_timer.hpp>

#include <ATen/xpu/XPUEvent.h>
#include <c10/core/DeviceGuard.h>

namespace c10d {
namespace {

const int kMilliSecondToNanosSecond = 1000000;

class XpuTimer : public Timer {
 private:
  c10::Device device;
  // at::xpu::XPUEvent(1) means enable_timing=true
  // at::xpu::XPUEvent() means enable_timing=false
  at::xpu::XPUEvent forward_start = at::xpu::XPUEvent();
  at::xpu::XPUEvent backward_compute_start = at::xpu::XPUEvent();
  at::xpu::XPUEvent backward_compute_end = at::xpu::XPUEvent();
  at::xpu::XPUEvent backward_comm_start = at::xpu::XPUEvent();
  at::xpu::XPUEvent backward_comm_end = at::xpu::XPUEvent();

  at::xpu::XPUEvent& getEvent(Event event) {
    switch (event) {
      case Event::kForwardStart:
        return forward_start;
      case Event::kBackwardComputeStart:
        return backward_compute_start;
      case Event::kBackwardComputeEnd:
        return backward_compute_end;
      case Event::kBackwardCommStart:
        return backward_comm_start;
      case Event::kBackwardCommEnd:
        return backward_comm_end;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
  }

 public:
  explicit XpuTimer(c10::Device dev) : device(dev) {}

  void record(Event event) override {
    // Parent class sets the host-side time
    Timer::record(event);
    c10::DeviceGuard g(device);
    getEvent(event).record();
  }

  std::optional<int64_t> measureDifference(Event start, Event end) override {
    c10::DeviceGuard g(device);
    at::xpu::XPUEvent& start_event = getEvent(start);
    at::xpu::XPUEvent& end_event = getEvent(end);

    if (!start_event.isCreated() || !end_event.isCreated()) {
      return std::nullopt;
    }

    start_event.synchronize();
    end_event.synchronize();

    // Event elapsed_time may cause stuck, disable it for now
    // float milliseconds = start_event.elapsed_time(end_event);
    float milliseconds = 0;

    if (milliseconds < 0) {
      return std::nullopt;
    }
    return int64_t(milliseconds * kMilliSecondToNanosSecond);
  }
};

C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kXPU, XpuTimer)

} // namespace
} // namespace c10d
