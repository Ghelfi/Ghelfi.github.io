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
honest, that alone is enough); 2) It dominates the industry and research landscape, so it’s kind of
the big deal.

# PyTorch and GPUs: The Basics
