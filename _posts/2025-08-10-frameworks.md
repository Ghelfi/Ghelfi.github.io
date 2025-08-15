---
layout: posts
title: 0. Deep Learning Frameworks
category: pytorch_internals
author_profile: false
sidebar: true
---

Star Ranking: ⭐

I mention [`PyTorch`](https://pytorch.org/) explicitly on the main page of this blog. One might
reasonably ask: *Why `PyTorch`, and not some other deep learning framework?* Or even, *Why pick a
specific framework at all?* The aim here is to dig into how things work under the hood, and choosing
a single framework gives us a concrete playground. The insights gained should transfer well to
others.

So..., why `PyTorch`? If I had written this blog ten years ago, would I have chosen differently?
Putting aside the minor temporal hiccup that `PyTorch` `v0.1.1` was released in April 2016, let’s
focus on the real question here: What drove the dynamics of development and adoption of deep
learning frameworks over the last decade?

# The Framework race

Over the past decade, the deep learning framework landscape has shifted dramatically—starting with
early pioneers like `Theano` and `Caffe`, which powered research breakthroughs in computer vision;
moving on to the first giants such as `TensorFlow`, and arriving at today’s dominant
community-driven contender, `PyTorch`.

Tracking the community adoption and impact of a deep learning framework is a challenge in
itself.<br> Should we measure its influence in research, for example, by counting the number of
papers published each year that use it? While sharing code alongside papers is the norm today, it
was far from standard a decade ago.<br> Another option is to track GitHub activity in public
repositories — issues, or pull requests? Here is the question. Each metric comes with its own
biases, from how requests are handled and internal development practices, to the scope and scale of
contributions. Below is a graph showing the number of pull requests per year for each repository. As
with any metric, it’s biased — but our goal here isn’t perfect precision, just a visualisation of
the overall trends (along the same lines, only a selection of frameworks is shown).

![The Rise and Fall of Deep Learning Frameworks](/assets/images/frameworks_prs_eras.png)

We can clearly identify three distinct eras:

- **The Precursor Era:** Only a few frameworks existed, mostly developed in academic settings.
- **The Framework War:** Many new contenders emerged, some backed by major companies — Google with
  TensorFlow, Facebook/Meta with PyTorch, Amazon with MXNet, and Baidu with Paddle.
- **PyTorch Domination:** PyTorch gains overwhelming community traction, effectively becoming the
  dominant framework. Paddle remains popular, particularly in Asia, but the overall landscape is
  largely shaped by PyTorch.

What could have triggered these shifts between eras? <br> The explosive rise of performance in
computer vision — and later in NLP — pushed large companies to develop their own frameworks to gain
a competitive edge, streamline internal projects, and accelerate research. This helps explain the
transition from the Precursor Era to the Framework War.

The second transition is less obvious. TensorFlow had an early advantage, offering a complete
solution from research to deployment, covering everything from large servers up to embedded devices
with TFLite. In technology, “winners take it all,” and once a full pipeline is built on a particular
framework, switching to another just to reach feature parity can be costly. This reinforces the
non-obvious nature of the second transition.

Two innovations, however, flipped the table: **eager execution** and **ONNX (Open Neural Network
Exchange)**. The oldies (or the curious ones exploring old research repositories) might remember
that TensorFlow 1 worked in two steps: first, you declared and built a compute graph with TensorFlow
operations; then, you fed this graph with data in a dedicated session (similar to how ONNX Runtime
works today). This approach made debugging painful, as inspecting internal data or controlling flow
was *cumbersome*—truth is, it was painful as hell. <br> **Eager execution** revolutionised the way
frameworks operate by running operations immediately, enabling direct inspection of data and
seamless integration of control flow. On top of that, the **ONNX format**, released in 2017,
established a standardised description for neural network operations. This effectively decoupled
training from inference: you could train a model with one framework and run it in another. The
separation encouraged frameworks to prioritise flexibility and easy debugging for
training—accelerating research and innovation—while allowing inference engines to be highly
optimised for ONNX models
