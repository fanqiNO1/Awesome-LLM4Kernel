# Awesome-LLM4Kernel

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Awesome LLM4Kernel](https://img.shields.io/badge/Awesome-LLM4Kernel-blue)](https://github.com/topics/awesome)

Awesome-LLM4Kernel: A curated list of papers with codes related to LLM-based kernel generation and optimization.

## 📚 Table of Contents

- [📝 By Generative Paradigm](#-by-generative-paradigm)
    - [⚡ Torch Related](#-torch-related)
    - [💻 C / C++ Related](#-c--c-related)
    - [🔤 Natural Language Related](#-natural-language-related)
    - [🧩 Other Paradigms](#-other-paradigms)
- [🎯 By Technical Focus](#-by-technical-focus)
    - [📊 Benchmarks](#-benchmarks)
    - [⚙️ Fine-tuning for Kernel Generation](#️-fine-tuning-for-kernel-generation)
    - [🔁 Self-Refinement & Iterative Optimization](#-self-refinement--iterative-optimization)
    - [🧩 Other Methods](#-other-methods)
- [📘 All Papers (Sorted By Date)](#-all-papers-sorted-by-date)
- [✨ Contributing](#-contributing)

---

## 📝 By Generative Paradigm

### ⚡ Torch Related

- [KernelBench](#202502-icml-2025-kernelbench-can-llms-write-efficient-gpu-kernels): Torch2CUDA, (Torch2Triton, Torch2CuTe, Torch2TileLang)
- [TritonBench](#202502-acl-findings-2025-tritonbench-benchmarking-large-language-model-capabilities-for-generating-triton-operators): Torch2Triton
- [KernelLLM](#202504-making-kernel-development-more-accessible-with-kernelllm): Torch2Triton
- [GPU Kernel Scientist](#202506-gpu-kernel-scientist-an-llm-driven-framework-for-iterative-kernel-optimization): Torch2HIP (AMD)
- [AutoTriton](#202507-autotriton-automatic-triton-programming-with-reinforcement-learning-in-llms): Torch2Triton
- [Kevin](#202507-kevin-multi-turn-rl-for-generating-cuda-kernels): Torch2CUDA
- [CUDA-L1](#202507-iclr-2026-cuda-l1-improving-cuda-optimization-via-contrastive-reinforcement-learning): Torch2CUDA
- [MultiKernelBench](#202507-multikernelbench-a-multi-platform-benchmark-for-kernel-generation): Torch2CUDA, Torch2AscendC (Huawei NPU), Torch2Pallas (Google TPU)
- [GEAK](#202507-geak-introducing-triton-kernel-ai-agent--evaluation-benchmarks): Torch2Triton
- [CudaLLM](#202508-cudallm-training-language-models-to-generate-high-performance-cuda-kernels): Torch2CUDA
- [Robust-KBench](#202509-towards-robust-agentic-cuda-kernel-benchmarking-verification-and-optimization): Torch2CUDA
- [ConCuR](#202510-concur-conciseness-makes-state-of-the-art-kernel-generation): Torch2CUDA
- [TritonRL](#202510-tritonrl-training-llms-to-think-and-code-triton-without-cheating): Torch2Triton
- [STARK](#202510-iclr-2026-stark-strategic-team-of-agents-for-refining-kernels): Torch2CUDA
- [TritonGym](#202510-tritongym-a-benchmark-for-agentic-llm-workflows-in-triton-gpu-code-generation): Torch2Triton
- [CudaForge](#202511-cudaforge-an-agent-framework-with-hardware-feedback-for-cuda-kernel-optimization): Torch2CUDA
- [PRAGMA](#202511-pragma-a-profiling-reasoned-multi-agent-framework-for-automatic-kernel-optimization): Torch2CUDA
- [KForge](#202511-kforge-program-synthesis-for-diverse-ai-hardware-accelerators): Torch2CUDA, Torch2MPS (Metal Performance Shaders, Apple)
- [KernelFalcon](#202511-kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents): Torch2CUDA
- [PIKE-B](#202511-optimizing-pytorch-inference-with-llm-based-multi-agent-systems): Torch2CUDA
- [QiMeng-Kernel](#202511-aaai-2026-qimeng-kernel-macro-thinking-micro-coding-paradigm-for-llm-based-high-performance-gpu-kernel-generation): Torch2CUDA
- [TritonForge](#202512-tritonforge-profiling-guided-framework-for-automated-triton-kernel-optimization): Torch2Triton
- [TritorX](#202512-agentic-operator-generation-for-ml-asics): Torch2Triton (Triton-MTIA: Meta Training and Inference Accelerator)
- [cuPilot](#202512-cupilot-a-strategy-coordinated-multi-agent-framework-for-cuda-kernel-evolution): Torch2CUDA
- [AKG kernel Agent](#202512-akg-kernel-agent-a-multi-agent-framework-for-cross-platform-kernel-synthesis): Torch2Triton, Torch2TileLang, Torch2CUDA
- [AscendCraft](#202601-ascendcraft-automatic-ascend-npu-kernel-generation-via-dsl-guided-transcompilation): Torch2AscendC (Huawei NPU)
- [Dr. Kernel](#202602-dr-kernel-reinforcement-learning-done-right-for-triton-kernel-generations): Torch2Triton
- [Makora](#202602-fine-tuning-gpt-5-for-gpu-kernel-generation): Torch2CUDA


### 💻 C / C++ Related

- [HPCTransCompile](#202506-hpctranscompile-an-ai-compiler-generated-dataset-for-high-performance-cuda-transpilation-and-llm-preliminary-exploration): CUDA2C / C++
- [QiMeng-MuPa](#202506-neurips-2025-qimeng-mupa-mutual-supervised-learning-for-sequential-to-parallel-code-translation): C++2CUDA

### 🔤 Natural Language Related

- [ComputeEval](#202504-computeeval-evaluating-large-language-models-for-cuda-code-generation): Text2CUDA
- [CUDA-LLM](#202506-cuda-llm-llms-can-write-efficient-cuda-kernels): Text2CUDA
- [PEAK](#202512-peak-a-performance-engineering-ai-assistant-for-gpu-kernels-powered-by-natural-language-transformations): Text2CUDA, Text2HIP (AMD), TextHLSL (High-Level Shader Language, Microsoft)
- [AscendKernelGen](#202601-ascendkernelgen-a-systematic-study-of-llm-based-kernel-generation-for-neural-processing-units): Text2AscendC (Huawei NPU)
- [KernelEvolve](#202512-kernelevolve-scaling-agentic-kernel-coding-for-heterogeneous-ai-accelerators-at-meta): Text2Triton
- [PACEvolve](202601-pacevolve-enabling-long-horizon-progress-aware-consistent-evolution): Text2CUDA

### 🧩 Other Paradigms

- [SwizzlePerf](#202508-swizzleperf-hardware-aware-llms-for-gpu-kernel-performance-optimization): Specific Kernels (`GEMM`, `fused element-wise`, `LayerNorm`, `Softmax`, `naive sparse matrix vector multiplication (SpMV)`, `transpose`, `Black-Scholes`, `finite-difference time-domain (FDTD) 2D`, `Smith-Waterman`, `Stencil 2D`)
- [Astra](#202509-astra-a-multi-agent-system-for-gpu-kernel-performance-optimization): CUDA2CUDA, Specific Kernels (`silu_and_mul`, `fused_add_rmsnorm`, `merge_attn_states_lse`)
- [RISC-V Kernels](#202509-evolution-of-kernels-automated-risc-v-kernel-optimization-with-large-language-models): RISC-V Kernels Optimization
- [EvoEngineer](#202510-evoengineer-mastering-automated-cuda-kernel-code-evolution-with-large-language-models): CUDA2CUDA
- [Performance Tool](#202510-integrating-performance-tools-in-model-reasoning-for-gpu-kernel-optimization): CUDA2CUDA
- [ReGraphT](#202510-from-large-to-small-transferring-cuda-optimization-expertise-via-reasoning-graph): CUDA2CUDA
- [SparseRL](#202510-iclr-2026-mastering-sparse-cuda-generation-through-pretrained-models-and-deep-reinforcement-learning): CUDA
- [MaxCode](#202510-maxcode-a-max-reward-reinforcement-learning-framework-for-automated-code-optimization): CUDA2CUDA
- [AccelOpt](#202511-accelOpt-a-self-improving-llm-agentic-system-for-ai-accelerator-kernel-optimization): NKI2NKI (Neuron Kernel Interface, AWS Trainium accelerator) 
- [CUDA-L2](#202512-cuda-l2-surpassing-cublas-performance-for-matrix-multiplication-through-reinforcement-learning): `Half-precision General Matrix Multiply (HGEMM)`
- [MEP-based LLM framework](#202512-gpu-kernel-optimization-beyond-full-builds-an-llm-framework-with-minimal-executable-programs): CUDA2CUDA
- [Two-Stage GPU Kernel Tuner](#202601-a-two-stage-gpu-kernel-tuner-combining-semantic-refactoring-and-search-based-optimization): CUDA2CUDA

## 🎯 By Technical Focus

### 📊 Benchmarks

- [KernelBench](#202502-icml-2025-kernelbench-can-llms-write-efficient-gpu-kernels): Torch2CUDA, (Torch2Triton, Torch2CuTe, Torch2TileLang)
- [TritonBench](#202502-acl-findings-2025-tritonbench-benchmarking-large-language-model-capabilities-for-generating-triton-operators): Torch2Triton
- [ComputeEval](#202504-computeeval-evaluating-large-language-models-for-cuda-code-generation): Text2CUDA
- [HPCTransEval & KernelBench_C](#202506-hpctranscompile-an-ai-compiler-generated-dataset-for-high-performance-cuda-transpilation-and-llm-preliminary-exploration): CUDA2C / C++
- [BabelTower](#202506-neurips-2025-qimeng-mupa-mutual-supervised-learning-for-sequential-to-parallel-code-translation): C++2CUDA
- [MultiKernelBench](#202507-multikernelbench-a-multi-platform-benchmark-for-kernel-generation): Torch2CUDA, Torch2AscendC (Huawei NPU), Torch2Pallas (Google TPU)
- [TritonGym](#202510-tritongym-a-benchmark-for-agentic-llm-workflows-in-triton-gpu-code-generation): Torch2Triton
- [FlashInfer-Bench](#202601-flashinfer-bench-building-the-virtuous-cycle-for-ai-driven-llm-systems): CUDA/Triton Optimization
- [CUDABench](#202603-cudabench-benchmarking-llms-for-text-to-cuda-generation): Text2CUDA
- [MobileKernelBench](#202603-iclr-workshop-2026-mobilekernelbench-can-llms-write-efficient-kernels-for-mobile-devices?):
- [MSKernelBench](#202603-making-llms-optimize-multi-scenario-cuda-kernels-like-experts): Text2CUDA

### ⚙️ Fine-tuning for Kernel Generation

- [KernelLLM](#202504-making-kernel-development-more-accessible-with-kernelllm): SFT
- [HPCTransCompile](#202506-hpctranscompile-an-ai-compiler-generated-dataset-for-high-performance-cuda-transpilation-and-llm-preliminary-exploration): SFT
- [QiMeng-MuPa](#202506-neurips-2025-qimeng-mupa-mutual-supervised-learning-for-sequential-to-parallel-code-translation): SFT
- [AutoTriton](#202507-autotriton-automatic-triton-programming-with-reinforcement-learning-in-llms): SFT + RL (GRPO)
- [Kevin](#202507-kevin-multi-turn-rl-for-generating-cuda-kernels): Multi-turn RL (GRPO)
- [CUDA-L1](#202507-iclr-2026-cuda-l1-improving-cuda-optimization-via-contrastive-reinforcement-learning): Contrastive RL
- [CudaLLM](#202508-cudallm-training-language-models-to-generate-high-performance-cuda-kernels): SFT + RL
- [ConCuR](#202510-concur-conciseness-makes-state-of-the-art-kernel-generation): SFT
- [TritonRL](#202510-tritonrl-training-llms-to-think-and-code-triton-without-cheating): SFT + RL (GRPO)
- [Performance Tool](#202510-integrating-performance-tools-in-model-reasoning-for-gpu-kernel-optimization): RL (GRPO)
- [SparseRL](#202510-iclr-2026-mastering-sparse-cuda-generation-through-pretrained-models-and-deep-reinforcement-learning): Pretrain + SFT + RL
- [QiMeng-Kernel](#202511-aaai-2026-qimeng-kernel-macro-thinking-micro-coding-paradigm-for-llm-based-high-performance-gpu-kernel-generation): RL (RLVR)
- [CUDA-L2](#202512-cuda-l2-surpassing-cublas-performance-for-matrix-multiplication-through-reinforcement-learning): RL
- [AscendKernelGen](#202601-ascendkernelgen-a-systematic-study-of-llm-based-kernel-generation-for-neural-processing-units): SFT + RL
- [Dr. Kernel](#202602-dr-kernel-reinforcement-learning-done-right-for-triton-kernel-generations): RL
- [Makora](#202602-fine-tuning-gpt-5-for-gpu-kernel-generation): RL

### 🔁 Self-Refinement & Iterative Optimization

- [CUDA-LLM](#202506-cuda-llm-llms-can-write-efficient-cuda-kernels): Single LLM
- [GPU Kernel Scientist](#202506-gpu-kernel-scientist-an-llm-driven-framework-for-iterative-kernel-optimization): Multiple LLMs (`Evolutionary Selector`, `Experimental Designer`, `Kernel Writer`)
- [GEAK](#202507-geak-introducing-triton-kernel-ai-agent--evaluation-benchmarks): Agents
- [SwizzlePerf](#202508-swizzleperf-hardware-aware-llms-for-gpu-kernel-performance-optimization): Single LLM to perform `swizzling`, FeedBack by L2 Hit Rate
- [Astra](#202509-astra-a-multi-agent-system-for-gpu-kernel-performance-optimization): Agents
- [RISC-V Kernels](#202509-evolution-of-kernels-automated-risc-v-kernel-optimization-with-large-language-models): Single LLM
- [RobustAgent](#202509-towards-robust-agentic-cuda-kernel-benchmarking-verification-and-optimization): Agents
- [EvoEngineer](#202510-evoengineer-mastering-automated-cuda-kernel-code-evolution-with-large-language-models): Single LLM
- [STARK](#202510-iclr-2026-stark-strategic-team-of-agents-for-refining-kernels): Agents
- [CudaForge](#202511-cudaforge-an-agent-framework-with-hardware-feedback-for-cuda-kernel-optimization): Agents
- [PRAGMA](#202511-pragma-a-profiling-reasoned-multi-agent-framework-for-automatic-kernel-optimization): Agents
- [KForge](#202511-kforge-program-synthesis-for-diverse-ai-hardware-accelerators): Agents
- [AccelOpt](#202511-accelOpt-a-self-improving-llm-agentic-system-for-ai-accelerator-kernel-optimization): Agents
- [KernelFalcon](#202511-kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents): Agents
- [PIKE-B](#202511-optimizing-pytorch-inference-with-llm-based-multi-agent-systems): Agents
- [TritonForge](#202512-tritonforge-profiling-guided-framework-for-automated-triton-kernel-optimization): Multiple LLMs (`Test Generation`, `Kernel Optimization`)
- [TritorX](#202512-agentic-operator-generation-for-ml-asics): Agents
- [cuPilot](#202512-cupilot-a-strategy-coordinated-multi-agent-framework-for-cuda-kernel-evolution): Agents
- [PEAK](#202512-peak-a-performance-engineering-ai-assistant-for-gpu-kernels-powered-by-natural-language-transformations): Single LLM (`Natural Language Transformations`)
- [MEP-based LLM framework](#202512-gpu-kernel-optimization-beyond-full-builds-an-llm-framework-with-minimal-executable-programs): Multiple LLMs
- [KernelEvolve](#202512-kernelevolve-scaling-agentic-kernel-coding-for-heterogeneous-ai-accelerators-at-meta): Agents
- [AKG kernel Agent](#202512-akg-kernel-agent-a-multi-agent-framework-for-cross-platform-kernel-synthesis): Agents
- [PACEvolve](202601-pacevolve-enabling-long-horizon-progress-aware-consistent-evolution): Single LLM
- [Two-Stage GPU Kernel Tuner](#202601-a-two-stage-gpu-kernel-tuner-combining-semantic-refactoring-and-search-based-optimization): Agents
- [AscendCraft](#202601-ascendcraft-automatic-ascend-npu-kernel-generation-via-dsl-guided-transcompilation): Multiple LLMs (`DSL Code Generation`, `Transcompilation`)
- [CUDAMaster](#202603-making-llms-optimize-multi-scenario-cuda-kernels-like-experts): Agents

### 🧩 Other Methods

- [ReGraphT](#202510-from-large-to-small-transferring-cuda-optimization-expertise-via-reasoning-graph): Reasoning Graph + Monte Carol Search
- [MaxCode](#202510-maxcode-a-max-reward-reinforcement-learning-framework-for-automated-code-optimization): Classical RL

---

## 📘 All Papers (Sorted By Date)

### (2025.02) [ICML 2025] KernelBench: Can LLMs Write Efficient GPU Kernels?

> 📃 [Paper](https://arxiv.org/abs/2502.10517)
>  
> 🛠️ [Code](https://github.com/ScalingIntelligence/KernelBench) ![Stars](https://img.shields.io/github/stars/ScalingIntelligence/KernelBench.svg)

### (2025.02) [ACL Findings 2025] TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators

> 📃 [Paper](https://arxiv.org/abs/2502.14752)
>  
> 🛠️ [Code](https://github.com/thunlp/TritonBench) ![Stars](https://img.shields.io/github/stars/thunlp/TritonBench.svg)

### (2025.04) ComputeEval: Evaluating Large Language Models for CUDA Code Generation

> 🛠️ [Code](https://github.com/NVIDIA/compute-eval) ![Stars](https://img.shields.io/github/stars/NVIDIA/compute-eval.svg)

### (2025.04) Making Kernel Development more accessible with KernelLLM

> 🤗 [Model](https://huggingface.co/facebook/KernelLLM)

### (2025.06) CUDA-LLM: LLMs Can Write Efficient CUDA Kernels

> 📃 [Paper](https://arxiv.org/abs/2506.09092)

### (2025.06) HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration

> 📃 [Paper](https://arxiv.org/abs/2506.10401)
>  
> 🛠️ [Code](https://github.com/PJLAB-CHIP/HPCTransCompile) ![Stars](https://img.shields.io/github/stars/PJLAB-CHIP/HPCTransCompile.svg)

### (2025.06) [NeurIPS 2025] QiMeng-MuPa: Mutual-Supervised Learning for Sequential-to-Parallel Code Translation

> 📃 [Paper](https://arxiv.org/abs/2506.11153)
>  
> 🛠️ [Code](https://github.com/QiMeng-IPRC/QiMeng-MuPa) ![Stars](https://img.shields.io/github/stars/QiMeng-IPRC/QiMeng-MuPa.svg)

### (2025.06) GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization

> 📃 [Paper](https://arxiv.org/abs/2506.20807)

### (2025.07) AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs

> 📃 [Paper](https://arxiv.org/abs/2507.05687)
>  
> 🛠️ [Code](https://github.com/AI9Stars/AutoTriton) ![Stars](https://img.shields.io/github/stars/AI9Stars/AutoTriton.svg)

### (2025.07) Kevin: Multi-Turn RL for Generating CUDA Kernels

> 📃 [Paper](https://arxiv.org/abs/2507.11948)

### (2025.07) [ICLR 2026] CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning

> 📃 [Paper](https://arxiv.org/abs/2507.14111)
>  
> 🛠️ [Code](https://github.com/deepreinforce-ai/CUDA-L1) ![Stars](https://img.shields.io/github/stars/deepreinforce-ai/CUDA-L1.svg)

### (2025.07) MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation

> 📃 [Paper](https://arxiv.org/abs/2507.17773)
>  
> 🛠️ [Code](https://github.com/wzzll123/MultiKernelBench) ![Stars](https://img.shields.io/github/stars/wzzll123/MultiKernelBench.svg)

### (2025.07) Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks

> 📃 [Paper](https://arxiv.org/abs/2507.23194)
>  
> 🛠️ [Code](https://github.com/AMD-AGI/GEAK-agent) ![Stars](https://img.shields.io/github/stars/AMD-AGI/GEAK-agent.svg)

### (2025.08) CudaLLM: Training Language Models to Generate High-Performance CUDA Kernels

> 🛠️ [Code](https://github.com/ByteDance-Seed/cudaLLM) ![Stars](https://img.shields.io/github/stars/ByteDance-Seed/cudaLLM.svg)

### (2025.08) SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization

> 📃 [Paper](https://arxiv.org/abs/2508.20258)

### (2025.09) Astra: A Multi-Agent System for GPU Kernel Performance Optimization

> 📃 [Paper](https://arxiv.org/abs/2509.07506)
>  
> 🛠️ [Code](https://github.com/Anjiang-Wei/Astra) ![Stars](https://img.shields.io/github/stars/Anjiang-Wei/Astra.svg)

### (2025.09) Evolution of Kernels: Automated RISC-V Kernel Optimization with Large Language Models

> 📃 [Paper](https://arxiv.org/abs/2509.14265)

### (2025.09) Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization

> 📃 [Paper](https://arxiv.org/abs/2509.14279)
>  
> 🛠️ [Code](https://github.com/SakanaAI/robust-kbench) ![Stars](https://img.shields.io/github/stars/SakanaAI/robust-kbench.svg)

### (2025.10) EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models

> 📃 [Paper](https://arxiv.org/abs/2510.03760)

### (2025.10) ConCuR: Conciseness Makes State-of-the-Art Kernel Generation

> 📃 [Paper](https://arxiv.org/abs/2510.07356)

### (2025.10) TritonRL: Training LLMs to Think and Code Triton Without Cheating

> 📃 [Paper](https://arxiv.org/abs/2510.17891)

### (2025.10) [ICLR 2026] STARK: Strategic Team of Agents for Refining Kernels

> 📃 [Paper](https://arxiv.org/abs/2510.16996)

### (2025.10) Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization

> 📃 [Paper](https://arxiv.org/abs/2510.17158)

### (2025.10) From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph

> 📃 [Paper](https://arxiv.org/abs/2510.19873)

### (2025.10) [ICLR 2026] Mastering Sparse CUDA Generation through Pretrained Models and Deep Reinforcement Learning

> 📃 [Paper](https://openreview.net/forum?id=VdLEaGPYWT)
>  
> 🛠️ [Code](https://github.com/QiWu-NCIC/SparseRL) ![Stars](https://img.shields.io/github/stars/QiWu-NCIC/SparseRL.svg)

### (2025.10) TritonGym: A Benchmark for Agentic LLM Workflows in Triton GPU Code Generation

> 📃 [Paper](https://openreview.net/forum?id=oaKd1fVgWc)

### (2025.10) MaxCode: A Max-Reward Reinforcement Learning Framework for Automated Code Optimization

> 📃 [Paper](https://openreview.net/pdf?id=E4RYoWcbl2)

### (2025.11) CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization

> 📃 [Paper](https://arxiv.org/abs/2511.01884)
>  
> 🛠️ [Code](https://github.com/OptimAI-Lab/CudaForge) ![Stars](https://img.shields.io/github/stars/OptimAI-Lab/CudaForge.svg)

### (2025.11) PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization

> 📃 [Paper](https://arxiv.org/abs/2511.06345)

### (2025.11) KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents

> 📃 [Blog](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/)
>  
> 🛠️ [Code](https://github.com/meta-pytorch/KernelAgent) ![Stars](https://img.shields.io/github/stars/meta-pytorch/KernelAgent.svg)

### (2025.11) KForge: Program Synthesis for Diverse AI Hardware Accelerators

> 📃 [Paper](https://arxiv.org/abs/2511.13274)

### (2025.11) [MLSys 2026] AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization

> 📃 [Paper](https://arxiv.org/abs/2511.15915)

### (2025.11) Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems

> 📃 [Paper](https://arxiv.org/abs/2511.16964)

### (2025.11) KernelBand: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit

> 📃 [Paper](https://arxiv.org/abs/2511.18868)

### (2025.11) [AAAI 2026] QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation

> 📃 [Paper](https://arxiv.org/abs/2511.20100)

### (2025.12) CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning

> 📃 [Paper](https://arxiv.org/abs/2512.02551)
>  
> 🛠️ [Code](https://github.com/deepreinforce-ai/CUDA-L2) ![Stars](https://img.shields.io/github/stars/deepreinforce-ai/CUDA-L2.svg)

### (2025.12) TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization

> 📃 [Paper](https://arxiv.org/abs/2512.09196)

### (2025.12) Agentic Operator Generation for ML ASICs

> 📃 [Paper](https://arxiv.org/abs/2512.10977)

### (2025.12) cuPilot: A Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution

> 📃 [Paper](https://arxiv.org/abs/2512.16465)
>  
> 🛠️ [Code](https://github.com/champloo2878/cuPilot-Kernels) ![Stars](https://img.shields.io/github/stars/champloo2878/cuPilot-Kernels.svg)

### (2025.12) PEAK: A Performance Engineering AI-Assistant for GPU Kernels Powered by Natural Language Transformations

> 📃 [Paper](https://arxiv.org/abs/2512.19018)

### (2025.12) GPU Kernel Optimization Beyond Full Builds: An LLM Framework with Minimal Executable Programs

> 📃 [Paper](https://arxiv.org/abs/2512.22147)

### (2025.12) KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

> 📃 [Paper](https://arxiv.org/abs/2512.23236)

### (2025.12) AKG kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis

> 📃 [Paper](https://arxiv.org/abs/2512.23424)

### (2026.01) FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems

> 📃 [Paper](https://arxiv.org/abs/2601.00227)

### (2026.01) AscendKernelGen: A Systematic Study of LLM-Based Kernel Generation for Neural Processing Units

> 📃 [Paper](https://arxiv.org/abs/2601.07160)

### (2026.01) PACEvolve: Enabling Long-Horizon Progress-Aware Consistent Evolution

> 📃 [Paper](https://arxiv.org/abs/2601.10657)

### (2026.01) A Two-Stage GPU Kernel Tuner Combining Semantic Refactoring and Search-Based Optimization

> 📃 [Paper](https://arxiv.org/abs/2601.12698)

### (2026.01) AscendCraft: Automatic Ascend NPU Kernel Generation via DSL-Guided Transcompilation

> 📃 [Paper](https://arxiv.org/abs/2601.22760)

### (2026.02) Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations

> 📃 [Paper](https://arxiv.org/abs/2602.05885)
>  
> 🛠️ [Code](https://github.com/hkust-nlp/KernelGYM) ![Stars](https://img.shields.io/github/stars/hkust-nlp/KernelGYM.svg)

### (2026.02) Fine-Tuning GPT-5 for GPU Kernel Generation

> 📃 [Paper](https://arxiv.org/abs/2602.11000)

### (2026.02) DICE: Diffusion Large Language Models Excel at Generating CUDA Kernels

> 📃 [Paper](https://arxiv.org/abs/2602.11715)

### (2026.02) OptiML: An End-to-End Framework for Program Synthesis and CUDA Kernel Optimization

> 📃 [Paper](https://arxiv.org/abs/2602.12305)

### (2026.02) KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning

> 📃 [Paper](https://arxiv.org/abs/2602.14293)

### (2026.02) K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model

> 📃 [Paper](https://arxiv.org/abs/2602.19128)
>  
> 🛠️ [Code](https://github.com/caoshiyi/K-Search) ![Stars](https://img.shields.io/github/stars/caoshiyi/K-Search.svg)

### (2026.02) CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation

> 📃 [Paper](https://arxiv.org/abs/2602.24286)
>  
> 🛠️ [Code](https://github.com/BytedTsinghua-SIA/CUDA-Agent) ![Stars](https://img.shields.io/github/stars/BytedTsinghua-SIA/CUDA-Agent.svg)

### (2026.03) CUDABench: Benchmarking LLMs for Text-to-CUDA Generation

> 📃 [Paper](https://arxiv.org/abs/2603.02236)
>  
> 🛠️ [Code](https://github.com/CUDA-Bench/CUDABench) ![Stars](https://img.shields.io/github/stars/CUDA-Bench/CUDABench.svg)

### (2026.03) StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning

> 📃 [Paper](https://arxiv.org/abs/2603.02637)

### (2026.03) [ICLR Workshop 2026] MobileKernelBench: Can LLMs Write Efficient Kernels for Mobile Devices?

> 📃 [Paper](https://openreview.net/pdf?id=bTsMfGz85R)

### (2026.03) Making LLMs Optimize Multi-Scenario CUDA Kernels Like Experts

> 📃 [Paper](https://arxiv.org/abs/2603.07169)

---

## ✨ Contributing

Welcome to star ⭐ and open an issue or PR to improve this repo!
