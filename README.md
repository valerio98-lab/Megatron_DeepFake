| Type checking | Linting | Try it on colab |
| :---: | :----: | :------: |
| [![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://img.shields.io/badge/mypy-checked-blue)| ![type checking: mypy](https://img.shields.io/badge/mypy-checked-blue)| <a target="_blank" href="https://colab.research.google.com/github/valerio98-lab/Megatron_DeepFake/blob/main/notebooks/train.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

# Introduction

This repo contains the code for a model that we decided to call `Megatron`.

## Abstract

Recent advancements in deepfake detection have demonstrated the utility of integrating depth information with RGB data to expose synthetic manipulations in static images.[[1]](#1)
Extending this approach, our work applies a novel model to video sequences, leveraging the temporal continuity and additional context offered by consecutive frames to enhance detection accuracy. We propose an advanced framework that utilizes face depth masks and RGB data concurrently, hypothesizing that dynamic sequences provide richer information than static frames alone.

Critically, we address limitations in existing RGB attention mechanisms by employing a cross-attention mechanism that processes informational embeddings extracted from both RGB and depth data.
This method allows for a more nuanced interplay between the modalities, focusing on salient features that are pivotal for identifying deepfakes.
Initial results suggest that this sophisticated attention mechanism significantly refines the detection process, offering promising directions for more robust deepfake recognition technologies.

## Dataset

## References

<a id="1">[1]</a>  [A guided-based approach for deepfake detection:RGB-depth integration via
 features fusion](https://www.sciencedirect.com/science/article/pii/S0167865524000990)
