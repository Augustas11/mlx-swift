// Copyright © 2026 Malibu. macprovider fork addition (#2338).
//
// Split synchronous eval for mlx-swift's process-wide eval lock. See
// include/mlx_eval_wait.h for the phase contract. Only public mlx core API is
// used; no upstream mlx or mlx-c file is modified.

#include <vector>

#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx_eval_wait.h"

namespace {

using mlx::core::array;
using mlx::core::Event;

struct Pending {
  // Outputs that were not yet available when the lock was released. Phase 3
  // marks them available; nothing else touches them outside the lock.
  std::vector<array> outputs;
  // Copies of their completion events. Copying an Event copies a shared_ptr to
  // the device event, so phase 2 can wait without reading any array field.
  std::vector<Event> events;
};

// The stream eval_impl would pick for `outputs` (transforms.cpp eval_impl).
mlx::core::Stream eval_stream(const std::vector<array>& outputs) {
  for (auto& o : outputs) {
    if (o.status() == array::Status::unscheduled && o.has_primitive()) {
      return o.primitive().stream();
    }
  }
  return mlx::core::default_stream(mlx::core::default_device());
}

} // namespace

extern "C" int
mlx_eval_wait_begin(mlx_eval_wait* res, const mlx_array* arrays, size_t num) {
  res->ctx = nullptr;
  std::vector<array> outputs;
  try {
    outputs.reserve(num);
    for (size_t i = 0; i < num; ++i) {
      outputs.push_back(mlx_array_get_(arrays[i]));
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  if (outputs.empty()) {
    // mlx::core::eval returns early on an empty list; keep that.
    return 0;
  }

  bool any_unscheduled = false;
  for (auto& o : outputs) {
    if (o.status() == array::Status::unscheduled) {
      any_unscheduled = true;
      break;
    }
  }

  if (any_unscheduled) {
    bool scheduled_async = false;
    try {
      if (eval_stream(outputs).device != mlx::core::Device::cpu) {
        mlx::core::async_eval(outputs);
        scheduled_async = true;
      }
    } catch (std::exception&) {
      // Not reported: the stock synchronous eval below reruns the same graph
      // and reports its own error, if it still has one (e.g. async_eval is
      // refused inside a graph transformation, eval is not).
    }
    if (!scheduled_async) {
      // CPU stream or a failed async schedule: the stock path, still under
      // the lock, ending with every output available.
      try {
        mlx::core::eval(outputs);
      } catch (std::exception& e) {
        mlx_error(e.what());
        return 1;
      }
      return 0;
    }
  }

  // Everything is scheduled. Collect what is still running.
  auto pending = new Pending();
  try {
    for (auto& o : outputs) {
      if (o.status() == array::Status::unscheduled) {
        throw std::runtime_error("[eval] output left unscheduled");
      }
      if (o.is_available()) {
        continue;
      }
      if (o.event().valid()) {
        pending->events.push_back(o.event());
        pending->outputs.push_back(o);
      } else {
        // Evaluated without an event and not available cannot happen
        // (is_available() accepts that state); wait locked like stock.
        o.wait();
      }
    }
  } catch (std::exception& e) {
    delete pending;
    mlx_error(e.what());
    return 1;
  }
  if (pending->outputs.empty()) {
    delete pending;
    return 0;
  }
  res->ctx = pending;
  return 0;
}

extern "C" int mlx_eval_wait_wait(const mlx_eval_wait w) {
  if (!w.ctx) {
    return 0;
  }
  try {
    for (auto& e : static_cast<Pending*>(w.ctx)->events) {
      e.wait();
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" void mlx_eval_wait_finish(const mlx_eval_wait w) {
  if (!w.ctx) {
    return;
  }
  for (auto& o : static_cast<Pending*>(w.ctx)->outputs) {
    try {
      // The event is signaled, so this detaches it and sets the status to
      // available: the same end state array::wait() leaves.
      o.is_available();
    } catch (std::exception&) {
      // A device error was already reported by mlx_eval_wait_wait.
    }
  }
}

extern "C" void mlx_eval_wait_free(mlx_eval_wait w) {
  delete static_cast<Pending*>(w.ctx);
}
