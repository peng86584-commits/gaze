<div align="center">

<!-- 项目 Logo -->
![GAZELOOM Logo](https://github.com/user-attachments/assets/8a09450b-f78c-4cad-a80b-371f41967d94)

<!-- 主标题 -->
# ⚡ GAZELOOM ⚡

<!-- 副标题 -->
### 3D Driver Gaze Estimation Framework

<!-- 描述 -->
A lightweight and robust driver gaze estimation system powered by self-supervised learning and geometry guidance.

---

</div>

## 🚀 About

**GAZELOOM** is a driver gaze estimation framework designed for intelligent traffic safety and human-vehicle interaction.  
By leveraging **multi-modal geometric guidance** and **self-supervised feature extraction**, it accurately predicts driver gaze points in 3D space.

- 🔹 **Lightweight Model**: Only 4.97M parameters, suitable for resource-constrained devices  
- 🔹 **High-Precision Estimation**: Joint prediction of head pose and eye movement  
- 🔹 **Real-time Performance**: Adapted for in-vehicle real-time inference  
- 🔹 **Scene Generalization**: Handles lighting changes, occlusions, and pose variations

---

## 🧠 Architecture Overview

The architecture of **GAZELOOM** is designed to efficiently estimate the 3D gaze points of the driver by integrating several key components:

1. **Camera Input → Face Landmark → Head Pose → Eye Gaze Vector**  
   The input from the camera is processed to extract face landmarks, head pose, and eye gaze vectors.
   
2. **Multi-modal Geometry Guidance**  
   This component integrates spatial geometric priors from different sensor modalities (e.g., facial features, head orientation, and gaze vector), enhancing the model's robustness and accuracy.

3. **Cross-modal Gating Attention**  
   Cross-modal attention mechanisms are applied to adaptively align semantic (e.g., gaze) and geometric information, optimizing the fusion of both inputs for better gaze prediction.

4. **3D Gaze Point Prediction**  
   Finally, the processed features are used to predict the 3D gaze point, which represents the driver's point of attention in the 3D space of the vehicle environment.

---

## 📊 Datasets & Results

Here are the performance metrics on key datasets:

| **Dataset**              | **AUC ↑** | **L2 ↓**  | **AP ↑**   |
|--------------------------|:---------:|:---------:|:---------:|
| **GazeFollow**            | **0.964** | **0.1028**| -         |
| **VideoAttentionTarget**  | **0.945** | **0.101** | **0.917** |

> The GazeLoom model achieves high performance across multiple benchmarks with **lightweight architecture**.

---

## 🧩 Keywords
**Gaze Estimation · Self-Supervised Learning · Multi-Modal Fusion · Geometry-Guided Vision**

---

<p align="center" style="color:#8DF7FF;font-size:18px;">
“Where the gaze goes, intelligence follows.”
</p>

---

## 📸 Images

<!-- 第一张图片 -->
![Image 2](https://github.com/user-attachments/assets/c5eebd49-0aae-43ed-8f98-006f6228114c)

<!-- 第二张图片 -->
![Image 3](https://github.com/user-attachments/assets/8662444a-d6ea-4255-92b8-175da69e1dc4)

<!-- 第三张图片 -->
![Image 4](https://github.com/user-attachments/assets/a5ced171-192a-4b7b-a8b3-b2b65519e4de)

<!-- 第四张图片 -->
![Image 1](https://github.com/user-attachments/assets/c5eebd49-0aae-43ed-8f98-006f6228114c)

---



