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

**Spoiler**: We’ll zoom in on CUDA only for two reasons: 1) It’s the one I actually know (let’s be
honest, that alone is enough), and 2) It dominates the industry and research landscape.

# PyTorch and GPUs: The Basics

PyTorch obviously doesn’t just throw tensors onto the GPU and hope they run faster. Instead, it
relies on **device-aware tensors**. Each tensor knows which device it belongs to, and operations are
executed based on that location. This design allows the device to be stored as an attribute of the
tensor class, while keeping the overall tensor interface consistent and device-agnostic.

```python
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

x = torch.randn(3, 3, device=device)
softmax = torch.softmax(x, dim=-1)
y = x + softmax
```

In the example above, the tensor is placed on the GPU if available, otherwise it falls back to the
CPU. Notice that the rest of the code remains identical regardless of the device. Intermediate
tensors (like the `softmax` tensor in our example) are automatically created on the same device as
the input.<br> At first glance, this feels like magic — are we really writing Python that runs
directly on the GPU? The reality is of course more nuanced. PyTorch builds a stream of operations
for each device:

- On the CPU, operations are executed synchronously: each op finishes before the next begins.
- On the GPU, operations are asynchronous: PyTorch enqueues work into a CUDA stream, and the GPU
  executes it in the background while Python code continues.

This design allows overlapping computations with data transfers and enables efficient scheduling of
kernels.<br> On a CPU, operations are executed one after the other, and the program blocks until
each step completes. On a GPU, things are more interesting: operations are enqueued asynchronously
onto a stream. The Python thread keeps running, while the GPU driver and runtime decide when kernels
and memory transfers actually execute.<br> This has three major consequences:

- Overlap of work – A kernel can run while data is being copied to or from GPU memory. The GPU has
  specialized hardware units (compute cores and copy engines) that can operate in parallel.
- Better throughput – By queuing many operations at once, the GPU always has work ready, reducing
  idle time.
- Fewer host-to-device (H2D) transfers – Since intermediate results are instantiated directly on the
  same device as their inputs, PyTorch avoids unnecessary CPU ↔ GPU copies. This not only reduces
  latency but also lowers PCIe bandwidth pressure, which is often a real bottleneck.

Internally, PyTorch routes operations through its dispatcher. C++ macros like `TORCH_LIBRARY_IMPL`
and `REGISTER_DISPATCH` register both CPU and CUDA kernels for each operation, and at runtime the
dispatcher picks the correct implementation based on the tensor’s device.
