<div align="center">

<!-- 项目 Logo -->

<!-- 主标题 -->
# ⚡ GazeLoom⚡
![PP](https://github.com/user-attachments/assets/3c4055b2-9ec1-40ee-9c79-b664945336d9)
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
  <img  width="301" height="232" alt="图片11" src="https://github.com/user-attachments/assets/0f2ecb2d-3cd2-41ab-81fa-27e61aafe383" />
  <img  width="301" height="232" alt="图片12" src="https://github.com/user-attachments/assets/f3e91f3b-5e87-46a1-829a-327f8e3a721d" />
  <img  width="301" height="232"alt="图片13" src="https://github.com/user-attachments/assets/5be57660-463a-4003-a2a8-b5b5d1d75549" />


  </a>
  <a href="https://github.com/user-attachments/assets/c5eebd49-0aae-43ed-8f98-006f6228114c" target="_blank">
    <img width="301" height="232"alt="图片5" src="https://github.com/user-attachments/assets/0fcd3faf-e1b1-49cd-8377-1191fd277ce4" />
    <img width="301" height="232" alt="图片6" src="https://github.com/user-attachments/assets/9732d414-9a6a-4257-a961-852fca559244" />
    <img width="301" height="232" alt="图片7" src="https://github.com/user-attachments/assets/1d4f8971-6ece-4c7a-b9c9-ca485f1a7730" />

  </a>
  <a href="https://github.com/user-attachments/assets/8662444a-d6ea-4255-92b8-175da69e1dc4" target="_blank">
    <img width="301" height="232" alt="图片8" src="https://github.com/user-attachments/assets/7548f741-8b97-4528-af2f-92d2100ccae1" />
    <img width="301" height="232" alt="图片9" src="https://github.com/user-attachments/assets/f996580c-a005-4c7a-b511-fe9c7584b7e4" />
    <img  width="301" height="232"alt="图片10" src="https://github.com/user-attachments/assets/e3bead59-4718-441f-b701-2b23fa40d6dc" />

  </a>
  <a href="https://github.com/user-attachments/assets/a5ced171-192a-4b7b-a8b3-b2b65519e4de" target="_blank">
  <img width="301" height="232" alt="图片1" src="https://github.com/user-attachments/assets/8c2f4bc4-eb1f-4303-a93b-be26cd17d0f9" />
  <img width="301" height="232" alt="图片2" src="https://github.com/user-attachments/assets/7302f26f-8360-4d99-9130-3180c2942869" />
  <img width="301" height="232" alt="图片3" src="https://github.com/user-attachments/assets/bb47e194-9054-4cb9-9dc5-e25220f233a8" />

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
