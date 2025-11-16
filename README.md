<div align="center">

<!-- 项目 Logo -->
<img src="https://github.com/user-attachments/assets/8a09450b-f78c-4cad-a80b-371f41967d94" 
     alt="GAZELOOM Logo" width="260" height="165" />

<!-- 主标题 -->
<h1 style="font-size:58px;color:#00FFFF;text-shadow:0 0 10px #00FFFF, 0 0 25px #0088FF;">
⚡ GAZELOOM ⚡
</h1>

<!-- 副标题 -->
<h3 style="color:#C0FFFF;">
三维驾驶员凝视点估计框架 · <em>3D Driver Gaze Estimation Framework</em>
</h3>

<!-- 简介 -->
<p style="color:#A0F0FF;font-size:16px;">
基于自监督学习与几何引导的轻量级驾驶员注意力建模系统  
<em>Lightweight & Robust Driver Gaze Estimation powered by Self-Supervised Vision</em>
</p>

---

</div>

## 🚀 About

**GAZELOOM** 是一个面向智能交通安全与人机共驾的驾驶员凝视估计框架。  
通过 **多模态几何引导** 与 **自监督特征提取**，实现驾驶员注视点的高精度三维预测。

- 🔹 **轻量化模型**：仅 4.97M 参数，适配资源受限设备  
- 🔹 **高精度估计**：头部姿态 + 眼动联合预测  
- 🔹 **实时性优先**：适配车载端实时推理  
- 🔹 **多场景泛化**：应对光照变化、遮挡与姿态多样性  

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


