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

## 📸 Visuals

<!-- 图片 2, 3, 4 一排展示，统一大小 -->
<div align="center">
  <a href="https://github.com/user-attachments/assets/c5eebd49-0aae-43ed-8f98-006f6228114c" target="_blank">
    <img width="300" height="228" alt="图片5" src="https://github.com/user-attachments/assets/0fcd3faf-e1b1-49cd-8377-1191fd277ce4" />
    <img width="300" height="228" alt="图片6" src="https://github.com/user-attachments/assets/9732d414-9a6a-4257-a961-852fca559244" />
  </a>
  <a href="https://github.com/user-attachments/assets/8662444a-d6ea-4255-92b8-175da69e1dc4" target="_blank">
    <img src="https://github.com/user-attachments/assets/8662444a-d6ea-4255-92b8-175da69e1dc4" alt="Image 3" width="250" height="250" style="border-radius: 10px; transition: transform 0.3s ease; margin-right: 10px;">
   <img src="https://github.com/user-attachments/assets/a5ced171-192a-4b7b-a8b3-b2b65519e4de" alt="Image 4" width="250" height="250" style="border-radius: 10px; transition: transform 0.3s ease;">

  </a>
  <a href="https://github.com/user-attachments/assets/a5ced171-192a-4b7b-a8b3-b2b65519e4de" target="_blank">
    <img src="https://github.com/user-attachments/assets/a5ced171-192a-4b7b-a8b3-b2b65519e4de" alt="Image 4" width="250" height="250" style="border-radius: 10px; transition: transform 0.3s ease;">
  </a>
</div>

---

## ✨ Key Features

- 🧠 **Geometry-Guided Learning** – Combines semantic and geometric priors for robust gaze estimation  
- ⚙️ **Self-Supervised Backbone** – Reduces dependency on labeled data  
- 🚗 **Driver-Centric Design** – Optimized for in-cabin and driving environments  
- ⚡ **Lightweight Deployment** – Only 4.97M parameters, real-time performance on edge devices  

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

## ⚙️ Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourname/GAZELOOM.git
cd GAZELOOM
pip install -r requirements.txt
