---
layout: posts
title: 1. GPU Acceleration with PyTorch
category: pytorch_internals
author_profile: false
sidebar: true
---

Star Ranking: ⭐⭐

Having explored eager execution, it’s time to uncover the next layer: how PyTorch orchestrates GPUs
behind the scenes.

# PyTorch and GPUs: The Basics

PyTorch obviously doesn’t just throw tensors onto the GPU and hope they run faster. Instead, it
relies on **device-aware tensors**. Each tensor knows which device it belongs to, and operations are
selected and executed based on that. This design allows the device to be stored as an attribute of
the tensor class, and keep the overall tensor interface consistent and device-agnostic.

```python
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

x = torch.randn(3, 3, device=device)
softmax = torch.softmax(x, dim=-1)
y = x + softmax
```

In this example, the tensor data is placed on the GPU if available, otherwise it falls back to the
CPU. Notice that the rest of the code remains identical regardless of the device. Intermediate
tensors (i.e. `softmax` in the example) are automatically created on the same device as the
input.<br> At first glance, this feels like magic — you write the same code for CPU or GPU exection.
But are we really writing Python that runs directly on the GPU? Of course not. PyTorch abstracts
this for us by using a dispatcher and building a stream of operations for each device. These streams
behave differently depending on the device:

- On the CPU, operations are executed sequentially and synchronously in the main process.
- On the GPU, operations are executed sequentially and **asynchronously**: PyTorch enqueues ops into
  CUDA streams (multiple CUDA streams could operate simultaneously), and the GPU executes them in
  the background while the Python process continues. This means you can access a tensor in the main
  process before data is actually computed (if computation is long enough). In this case, Pytorch
  has guard system that triggers a synchronisation of the CUDA stream to make sure data is
  available.

This design of the GPU operation orchestration allows the overlap between compute and data
transfer.<br> This has three major consequences:

- Task overlap – A kernel can run while data is being copied to or from GPU memory. The GPU has
  specialized hardware units (compute cores and copy engines) that can operate in parallel.
- Higher throughput – By queuing many operations at once, the GPU always has work ready, reducing
  idle time.
- Fewer host-to-device (H2D) transfers – Since intermediate results are instantiated directly on the
  same device as their inputs, PyTorch avoids unnecessary CPU ↔ GPU copies. This not only reduces
  latency but also lowers PCIe bandwidth pressure, which is, nowdays, a common bottleneck.

# Dispatch and Fall back

PyTorch uses a dispatch system to decide which implementation of an operation it should use given
the tensor’s device, dtype, layout, and other properties.

The ATen dispatcher maps high-level operator (such as `aten::softmax`) to kernel implementation
based on a set of dispatch keys attached to the input tensors. When the dispatcher sees these keys,
it prioritises the CUDA implementation registered for `aten::softmax`. If no CUDA kernel is
available, PyTorch does not immediately fail. Instead, it falls back through a well-defined
hierarchy:

- Backend-specific kernel (e.g. CUDA)
- Generic kernel (device-agnostic or composite)
- Ultimately, an error if no valid implementation exists

This fallback mechanism is crucial. Many operators are implemented once using other ATen ops
(so-called composite operators). These composite implementations automatically work on any device
that supports the underlying primitives, which drastically reduces the amount of backend-specific
code required.
