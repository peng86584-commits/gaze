<div align="center">

<!-- 项目 Logo -->
<img src="https://github.com/user-attachments/assets/8a09450b-f78c-4cad-a80b-371f41967d94" 
     alt="GAZELOOM Logo" width="260" height="165" />

<!-- 主标题 -->
<h1 style="font-size:58px;color:#00FFFF;text-shadow:0 0 10px #00FFFF, 0 0 25px #0088FF;">
⚡ GazeLoom ⚡
</h1>

<!-- 副标题 -->
<h3 style="color:#C0FFFF;">
3D Driver Gaze Estimation Framework
</h3>

<!-- 描述 -->
<p style="color:#A0F0FF;font-size:16px;">
A lightweight and robust driver gaze estimation system powered by self-supervised learning and geometry guidance.
</p>

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


