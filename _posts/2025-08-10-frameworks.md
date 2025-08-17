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

Over the past decade, the deep learning framework landscape has shifted dramatically — starting with
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
  `TensorFlow`, Facebook/Meta with `PyTorch`, Amazon with `MXNet`, and Baidu with `Paddle`.
- **`PyTorch` Domination:** `PyTorch` gains overwhelming community traction, effectively becoming
  the dominant framework. Paddle remains popular, particularly in Asia, but the overall landscape is
  largely shaped by `PyTorch`.

What could have triggered these shifts between eras? <br> The explosive rise of performance in
computer vision — and later in NLP — pushed large companies to develop their own frameworks to gain
a competitive edge, streamline internal projects, and accelerate research. This helps explain the
transition from the Precursor Era to the Framework War.

The second transition is less obvious. `TensorFlow` had an early advantage, offering a complete
solution from research to deployment, covering everything from large servers up to embedded devices
with TFLite. In technology, *“winners take it all,”* and once a full pipeline is built on a
particular framework, switching to another just to reach feature parity can be costly. This
reinforces the non-obvious nature of the second transition while `TensorFlow` had a slight edge on
other competitors.

# Breaking Points in the Framework Race

Two innovations flipped the table, and dramatically reshaped the deep learning landscape: **eager
execution** and **ONNX (Open Neural Network Exchange) format**. <br> These breakthroughs addressed
two of the most painful bottlenecks in the early frameworks: the rigidity of computation graphs and
the lack of interoperability between training and deployment. Together, they marked the turning
point in the *Framework War*.

## Eager Execution: From Graphs to Pythonic Code

The oldies (or the curious ones exploring old research repositories) might remember that
`TensorFlow` 1 forced researchers to build static computation graphs. You had to:

1. Define your entire model as a symbolic graph.
2. Feed data into that graph via a Session.
3. Extract results at the end.

This made debugging extremely frustrating: intermediate values were hidden inside the graph, and
control flow was cumbersome — *read painful as hell*. <br> **Eager execution** flipped this
paradigm. In `PyTorch`, operations run immediately, just like Python code. The computation graph is
built dynamically as you execute operations, enabling intuitive debugging and rapid iteration.

To illustrate the impact of eager execution, let’s imagine a very small "network":

- It takes a single float input.
- It has a trainable weight W.
- It repeatedly multiplies the value by W inside a loop (say 5 iterations).
- it outputs the final output after the loop.

In addition, we want to inspect the intermediate values at each step — a common need when debugging
larger models (e.g., to check activations, gradients, or stability during training). Below are two
snippets doing this: one in static graph mode (TensorFlow 1.x), and one in eager execution mode
(PyTorch).

```python
# TensorFlow 1.x (static graph)
import tensorflow as tf

# Input placeholder
x = tf.placeholder(tf.float32, shape=[])

# Parameters
W = tf.Variable(2.0)

# The body of the for loop
def body(i: tf.Tensor, val: tf.Tensor, logs: tf.TensorArray):
    new_val = val * W
    logs = logs.write(i, new_val)
    return i + 1, new_val, logs

# The stoping condition of the fot loop
def cond(i: tf.Tensor, val: tf.Tensor, logs: tf.TensorArray):
    return i < 5

i0 = tf.constant(0)
logs = tf.TensorArray(dtype=tf.float32, size=5)

_, final_val, logs = tf.while_loop(cond, body, [i0, x, logs])

with tf.Session() as sess:
    # We need to Initialize the eventual weights (here, W)
    sess.run(tf.global_variables_initializer())
    # We need to specify the logs debug variable as output of our graph
    output, debug_vals = sess.run([final_val, logs.stack()], feed_dict={x: 3.0})
    print("Intermediate results:", debug_vals)
    print("Final result:", output)
```

```python
# PyTorch (eager execution)
import torch

# Input and parameters
x: torch.Tensor = torch.tensor(3.0)
W: torch.Tensor = torch.tensor(2.0)

logs: list[torch.Tensor] = []
for i in range(5):
    x = x * W
    logs.append(x)

print("Intermediate results:", logs)
print("Final result:", x)
```

These two snippets clearly illustrate the difference in developer experience. With eager execution,
the code reads like plain Python: loops, arithmetic, and logging happen naturally, and you can
inspect intermediate results directly. There’s no need for placeholders, sessions, or extra graph
machinery. This natural expressiveness makes experimentation, debugging, and iterative design far
faster and more intuitive. By lowering the friction between ideas and implementation, eager
execution has had a huge impact on innovation speed, enabling researchers and engineers to prototype
new architectures and training techniques much more efficiently than ever before.

## ONNX: Decoupling Training from Inference

The second breakthrough came in 2017 with the introduction of the ONNX format, a common standard for
representing neural networks. Before ONNX, frameworks formed silos: models trained in one framework
were locked into its ecosystem for deployment. <br> ONNX broke down these walls by providing a
portable, framework-agnostic representation of models. <br> This allowed researchers to:

- Train models in PyTorch for maximum flexibility.
- Export them to ONNX.
- Deploy them on optimized inference engines (ONNX Runtime, TensorRT, OpenVINO, etc.).

It also helped speed up the development of specific inference hardware since the standards brought
by ONNX allowed to support on input format.

The second breakthrough came in 2017 with the introduction of the ONNX format, a common standard for
representing neural networks. Before ONNX, frameworks were mostly siloed: models trained in one
framework were effectively locked into its ecosystem for deployment.<br> ONNX broke down these walls
by providing a portable, framework-agnostic representation of models.<br> This enabled researchers
to:

- Train models in PyTorch (or another framework) for maximum flexibility.
- Export them to ONNX.
- Deploy them on highly optimized inference engines such as ONNX Runtime, TensorRT, or OpenVINO.

ONNX also accelerated the development of specialized inference hardware: the standardization of
model formats meant that hardware vendors could optimize their platforms for a common input format,
reducing friction and improving performance across the ecosystem.

# So... PyTorch

`PyTorch`’s early leap into eager execution and the ONNX release gave it massive momentum —
something `TensorFlow` 2.0 (which finally enabled eager execution) couldn’t fully reverse. In the
end, picking `PyTorch` for this blog feels not just natural, but objectively the best choice today.
Ten years ago, you’d have read me grumbling about building graphs from protobuf definitions…

One might wonder if moving from a full pre-defined graph to eager execution could hurt compute
speed, especially on GPUs. Back then, GPUs and CUDA kernels weren’t as performant as today, but the
productivity and flexibility gains from eager execution far outweighed any slowdown. Fast-forward to
today, and with model sizes exploding, it’s a valid concern again.<br> Ironically, `PyTorch` is now
reintroducing partial graphs under the hood via model compilation. To really understand why — and
how it works — we first need to dig into how `PyTorch` handles GPUs. Coincidentally, that’s exactly
what the next post is all about…
