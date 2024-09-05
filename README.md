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

## Methodology

## Dataset

| Original | Deepfake | Face2Face |
| :---: | :----: | :------: |
| ![original](./assets/original_sample.gif)| ![deepfake](.assets/deepfake_sample.gif) | ![face2face](.assets/face2face.gif) |

Our study utilizes the FaceForensics++ dataset [[2]], which is widely recognized for its comprehensive set of video sequences designed explicitly for the training and evaluation of deepfake detection models.
The dataset includes an original folder and five additional folders that contain samples generated through various deepfake techniques.
Due to hardware constraints, we limited our focus to two specific types of deepfake generation techniques and applied a unique compression algorithm, differing from the dataset's default, to address computational limitations and explore the impact of compression artifacts on model performance.

### Data preparation

The data preparation process is integral to ensuring the quality and efficacy of the training regimen for deepfake detection. Our methodology unfolds through several meticulous steps:

1. Frame Extraction: Each video from the FaceForensics++ dataset is dissected into individual frames. This granular breakdown facilitates detailed analysis of each moment captured in the video, allowing for frame-specific deepfake detection.

2. Face Detection: We employ the dlib [[3]] library’s face detection method, which utilizes Histogram of Oriented Gradients (HOG) coupled with a Support Vector Machine (SVM). This method is preferred for its expediency and effectiveness, providing a robust solution for rapidly isolating faces from varied backgrounds and orientations.

3. Data Augmentation: To enhance the robustness of our model against diverse manipulations and increase the dataset size, we implement random transformations on the detected faces. This approach presumes the randomness of transformations to artificially expand our dataset, effectively multiplying the number of training samples by the number of transformations applied.

4. Depth Estimation: Subsequent to acquiring RGB images of faces, we generate corresponding depth masks using the ‘depth_anything’ model. This step introduces another dimension of data that our model can utilize to discern authentic from manipulated content.

5. Tensor Creation: To manage the variability in facial dimensions across frames, we opt for padding rather than resizing. Padding helps maintain the integrity of the face data without introducing geometric distortions that resizing might cause. This step ensures that all input tensors fed into the neural network are of uniform size.

6. Feature Extraction via RepVit: Citing another paper, AudioLM[[5]],
    > The key underlying mechanism of the best of these models (transformers) is self-attention,  which is suitable for
    > modeling rich and complex long-range dependencies but,  in the standard form, has a computational cost that grows
    > quadratically with the length of the input sequences. This cost is acceptable for sequences of up to 1000 tokens, however,
    > it prevents modeling natural signals in their raw form (for example, modeling a 512 × 512 image at the pixel level).
    > While several works have explored efficient alternatives to self attention, another solution to this scaling problem
    > is to work with mappings of the natural signals to a compact, discrete representation space.
    In particular we retrive a discrete rapresentation of both rgb frames and depth frames.
    By leveraging the RepVit model [[6]] we extract embeddings for subsequent processing.
    This adaptation not only accelerates the computation but also maintains the efficacy of the model under varying input sizes.

This methodology, from frame extraction to embedding generation, is designed to capture a comprehensive spectrum of features that are essential for accurate deepfake detection, utilizing both spatial and temporal data efficiently.

### Optimizations

In addressing the challenges posed by the computational constraints of Google Colab and the extensive requirements of our deepfake detection model, we implemented several key optimizations to enhance training efficiency and manage data processing effectively.

#### Embedding caching

The generation and processing of data, particularly with the augmented size due to transformations, introduced a significant bottleneck in the performance of our training processes.
A critical issue was the repeated computation of embeddings for identical input frames across different training iterations.
To resolve this, we adopted a strategy of caching the RepVit embeddings.
By storing these embeddings post-calculation, we drastically reduced the need for redundant computations.
The storage footprint for these embeddings is minimal, with a tensor comprising 128 elements, each representing 20 frames with an embedding dimension of 384, occupying approximately 5 MB.
This approach not only conserves memory but also accelerates the training phase by eliminating repetitive processing tasks.

#### Sequence Integrity and Batch Consistency

Our model assumes that video frames are sequential, which presents a unique challenge when certain frames fail to detect a face using dlib, or when such frames yield errors with the 'depth_anything' model.
The absence or faultiness of expected data in certain frames can disrupt the sequence integrity and affect batch processing.
Having diverse elements in a batch during training, especially in machine learning contexts like deepfake detection or other image processing tasks, is crucial for several reasons:

- Generalization: A diverse batch ensures that the model encounters a wide range of data variations (e.g., different lighting conditions, ages, ethnicities, expressions in faces). This diversity helps the model to generalize better to unseen data rather than just memorizing specific examples.
- Avoiding Bias: If the training batches repeatedly contain similar types of data, the model might develop a bias towards those features. For instance, if a facial recognition model is mostly trained on images of people from a single ethnicity, it might perform poorly on other ethnicities. Diverse batches help in reducing this risk.
- Robustness: Exposure to a wide variety of data during training can enhance the model’s robustness to noise, distortions, or variations in real-world scenarios. This is particularly important in applications like deepfake detection where subtle cues and differences need to be discerned accurately.
- Effective Learning: Diverse batches ensure that each update of the model's weights during training is informative and represents the overall distribution of the data. This can lead to more effective and efficient learning, preventing the model from overfitting to a narrow subset of the data.
- Balanced Learning: When training on imbalanced datasets, ensuring diversity in each batch can help mitigate the dominance of the majority class by giving adequate representation to minority classes within each training step.
To address this, we implemented an aggregation technique that ensures the maintenance of the required batch size, even when some elements within the data are invalid or absent. This method involves strategically filling in or ignoring deficient frames to preserve the continuous flow and consistency of data batches.

#### Impact on training time

The combined effect of embedding caching and batch consistency optimization has led to a significant reduction in training time.
By streamlining the data preparation and embedding generation processes, and ensuring consistent batch processing, we have achieved a more efficient training cycle.
This allows for quicker iterations and adaptations of the model, enhancing our ability to refine and improve detection accuracy within the constraints of our computational environment.

### Transone

TODO: Valerio

## Experiments

TODO: Valerio

## Model Selection Rationale

### Efficiency Requirements

The project’s overarching goal was to create a deepfake detection system that is not only effective but also efficient, adhering to stringent requirements concerning memory use, processing speed, and adaptability to constrained environments such as mobile devices.
These requirements dictated our choice of models, influencing both the architecture selection and the subsequent optimization strategies.

### Model Choice and Adaptation

Mobile-Friendly Models: The primary consideration was selecting models that are inherently designed for efficiency, particularly suitable for deployment in mobile environments where computational resources are limited. This requirement led us to choose models with proven efficiency in such settings.

### Efficiency Techniques

HW/Algorithm Co-Design: Considering the hardware limitations typical of mobile environments, we focused on HW/Algorithm co-design.
This approach ensures that the chosen models not only fit within the computational budgets of mobile devices but also exploit the specific hardware capabilities effectively. The RepVit model, with its simplified and compact architecture, aligns well with this strategy, optimizing both memory usage and processing speed.

Knowledge Distillation: To further enhance efficiency, we applied knowledge distillation techniques during the training process. This method involves training a smaller, more compact "student" model to replicate the performance of a larger "teacher" model. By distilling the knowledge from complex models into a more manageable form, we maintain high accuracy while significantly reducing the computational burden.

Memory Efficiency: In addition to model selection and design adaptations, memory efficiency was a critical aspect. By caching the embeddings generated by RepVit, we drastically reduced redundant computations and minimized memory usage, which is paramount in a mobile setting where RAM and storage are limited.

## References

<a id="1">[1]</a>  [A guided-based approach for deepfake detection:RGB-depth integration via
 features fusion](https://www.sciencedirect.com/science/article/pii/S0167865524000990)
<a id ="2">[2]</a> [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1901.08971)
<a id ="3">[3]</a> [Dlib: a modern C++ toolkit containing machine learning algorithms](http://dlib.net/)
<a id ="4">[4]</a> [Depth Anything V2](https://depth-anything-v2.github.io/)
<a id ="5">[5]</a> [AudioLM: a LAnguage modeling approach to audio generation](https://arxiv.org/pdf/2209.03143)
<a id ="6">[6]</a> [RepVit](https://arxiv.org/pdf/2307.09283)
