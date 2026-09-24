# Distributed With PyTorch

A hands-on repository for learning **distributed training and distributed inference with PyTorch**.

## Learning Path

```text
PyTorch
   ↓
Distributed Training
   ↓
Distributed Inference
```

### 1. PyTorch Fundamentals

Starting with this PyTorch refresher as I've spent most of my time writing CUDA C++ kernels:

* [PyTorch Complete Course](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)

Refreshing tensors, autograd, neural networks, training loops, datasets, optimization, and PyTorch's programming model before moving into distributed workloads.

### 2. Distributed Training

Primary resource:

* [Building a Distributed Training Framework from First Principles — Umar Jamil](https://www.youtube.com/watch?v=XoGvCBRnwLs)

Topics include:

* Process groups and collective communication
* Distributed autograd and DDP
* Data parallelism
* Pipeline parallelism
* FSDP
* Tensor parallelism
* Context parallelism
* Device meshes
* Mixture-of-Experts
* Expert parallelism
* All-to-All communication

The course builds these concepts from first principles and combines the different forms of parallelism into a working distributed training framework.

### 3. Distributed Inference

Topics to study:

* Prefill and decode
* KV cache
* Continuous batching
* Tensor parallelism
* Pipeline parallelism
* Expert parallelism
* Distributed serving
* Scheduling
* Latency and throughput
* Multi-GPU and multi-node inference

### Papers

A collection of papers for deeper study:

* [Distributed Training & Inference Papers](https://www.alphaxiv.org/shared/folder/019de088-28f7-7f02-acd4-c22459fe153e)

Selected papers from the learning material include:

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [GShard](https://arxiv.org/abs/2006.16668)
* [GPipe](https://arxiv.org/abs/1811.06965)
* [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
* [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
* [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
* [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

## Goal

Understand how modern workloads scale from **one process and one model** to **large distributed training and inference systems**.


