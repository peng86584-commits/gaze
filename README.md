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
```text
Camera → Face Landmark → Head Pose → Eye Gaze Vector 
        ↓
   Multi-modal Geometry Guidance
        ↓
   Cross-modal Gating Attention
        ↓
   3D Gaze Point Prediction
📊 Datasets & Results
| Dataset              |   AUC ↑   |    L2 ↓    |    AP ↑   |
| -------------------- | :-------: | :--------: | :-------: |
| GazeFollow           | **0.964** | **0.1028** |     -     |
| VideoAttentionTarget | **0.945** |  **0.101** | **0.917** |
🧩 Keywords
Gaze Estimation · Self-Supervised Learning · Multi-Modal Fusion · Geometry-Guided Vision

<p align="center" style="color:#8DF7FF;font-size:18px;"> “Where the gaze goes, intelligence follows.” </p>
<!-- 第一张图片 -->


<!-- 第二张图片 -->


<!-- 第三张图片 -->


<!-- 第四张图片 -->




