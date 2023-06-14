# Research on Iterative Magnitude Pruning for Depth Estimation Models

This repository contains the code and resources related to the research conducted on the application of iterative magnitude pruning for task-agnostic pre-trained models in the context of depth estimation. The research is based on the principles of the Lottery Ticket Hypothesis proposed by Jonathan Frankle.

## Storyline

### 1.1 Why is it interesting?

In the field of depth estimation models, task-specific fine-tuning on pre-trained models can be computationally expensive. This research explores the idea that pre-trained task-agnostic computer vision models, when subjected to iterative magnitude pruning, can maintain their downstream transferability for classification, detection, and segmentation tasks. However, the impact of pruning on the downstream transferability for the depth estimation task remains unexplored.

### 1.2 Current approach

The current approach to transfer learning for depth estimation involves three stages: pre-training a task-agnostic model, fine-tuning it for the specific task, and distilling a smaller task-specific model through pruning.

### 1.3 What is missing?

The existing methodology lacks a more efficient approach to fine-tuning on deep and wide task-specific models. The task-specific distillation (pruning) is performed after the fine-tuning process, which adds to the overall computational cost.

### 1.4 Proposed solution

To address the computational cost associated with fine-tuning, our research proposes the adoption of iterative magnitude pruning for the task-agnostic model before the fine-tuning process for the depth estimation task. This approach aims to significantly reduce the computational overhead while maintaining the model's performance.

### 1.5 Experimental questions

The research seeks to answer the following experimental questions:

1. Can we aggressively trim down the complexity of pre-trained models without damaging their downstream transferability on the depth estimation task?
