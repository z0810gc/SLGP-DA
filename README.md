# SLGP-DA: Structure-Prior Guided Dual-Path X-ray Data Augmentation (SLMY + GPPL)

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository presents the implementation of **SLGP-DA**, a **Structure-Prior Guided Dual-Path X-ray Data Augmentation** framework for rare defect learning in industrial DR inspection of power transmission compression fittings. The framework couples **physical imaging priors** with **generative AI**, and provides an end-to-end pipeline covering:

- **SLMY branch (Fracture synthesis)**: Stable Diffusion + LoRA with **mask-constrained local synthesis** and **YOLO-based posterior filtering**.
- **GPPL branch (Cavity expansion)**: **PCA axial prior** + **half-space constraint** + **axial stepwise growth** with **photometric alignment** and **boundary feather blending**.
- **Boundary continuity evaluation**: ΔG (boundary gradient difference) and ΔC (local contrast difference) for clean vs augmented consistency assessment.
- **YOLO11m-OBB full chain**: downstream OBB detection and a Qt-based visualization/grading tool.

> Note: Industrial DR datasets may be restricted and are NOT included by default. This repository provides reproducible code structure, configuration conventions, and usage guidance for academic verification.

---

## 📊 Architecture Overview

### ▶ SLGP-DA Dual-Path Framework (Fig. 5)
![Fig5](RE_IMAGE/Fig5.png)

> Figure 5: Overall structure-prior guided dual-path augmentation framework (SLMY for fracture synthesis, GPPL for cavity expansion) and its integration with downstream detection.

---

### ▶ SLMY Branch: Mask-Constrained Local Synthesis (Fig. 10–11)
![Fig10](RE_IMAGE/Fig10.png)

> Figure 10: Mask-constrained local synthesis mechanism in the latent space.

![Fig11](RE_IMAGE/Fig11.png)

> Figure 11: SLMY pipeline including LoRA prior injection, mask-guided synthesis, and YOLO-based posterior screening.

---

### ▶ GPPL Branch: Axial Prior & Half-Space Constrained Expansion (Fig. 12)
![Fig12](RE_IMAGE/Fig12.png)

> Figure 12: GPPL workflow including segmentation, PCA axis estimation, half-space constraint, stepwise axial expansion, photometric alignment, and blending.

---

### ▶ Dual-Branch Qualitative Results (Fig. 14–15)

<table>
  <tr>
    <td align="center">
      <img src="RE_IMAGE/Fig14(a).png" alt="Fig14(a)" width="480">
      <br><b>Fig. 14(a)</b>
    </td>
    <td align="center">
      <img src="RE_IMAGE/Fig14(b).png" alt="Fig14(b)" width="480">
      <br><b>Fig. 14(b)</b>
    </td>
  </tr>
</table>

> Figure 14: Representative qualitative results of dual-path augmentation. (a) [caption aligned with the paper]. (b) [caption aligned with the paper].

![Fig15](RE_IMAGE/Fig15.png)

> Figure 15: Additional qualitative results demonstrating geometric and photometric consistency.

---

## 🧰 System Requirements

| Component        | Version / Notes |
|------------------|------------------|
| Operating System | Linux / Windows (Linux recommended for diffusion workloads) |
| Python           | 3.8+ (3.10 recommended) |
| Diffusion (SLMY) | diffusers + transformers + accelerate + torch (GPU recommended) |
| Perception/Detection | ultralytics (YOLO11 series: Det / Seg / OBB as used in the paper) |
| Image Processing | OpenCV, NumPy, Pillow |
| GUI              | PyQt6 (for the OBB visualization/grading tool) |
| Optional Plotting | Matplotlib (for GPPL histogram/diagnostics if enabled) |

---

## 🚀 Guide

### 1️⃣ Data Conventions

This repository supports two types of data conventions:

**(A) SLMY input (mask-constrained synthesis)**
- Each sample should provide a **base DR image** and a corresponding **binary mask** indicating candidate synthesis regions.
- If using Labelme-style exports, group each sample into a folder with consistent naming (e.g., `xxx_json/` containing `img.*` and `label.*`).

**(B) GPPL input**
- Provide seed images containing cavity/under-insertion regions.
- Segmentation outputs (steel + cavity masks) are used to estimate PCA axis and apply constrained axial growth.

**(C) ΔG/ΔC evaluation input (naming alignment)**
- Masks are indexed by a leading integer ID (e.g., `1_mask_used.png`).
- Clean images share the same leading ID (e.g., `1.png`).
- Generated images also share the same leading ID (e.g., `1_*.png`).
- This ensures “same-location” evaluation between clean and augmented samples.

**(D) YOLO11m-OBB downstream dataset**
- Standard YOLO dataset structure is recommended (train/val/test splits).
- OBB label format should match your chosen YOLO11 implementation.

---

### 2️⃣ Run the Core Pipelines

#### 2.1 SLMY (Fracture Synthesis)
- Load a Stable Diffusion base model with LoRA weights.
- Perform mask-constrained local synthesis for each candidate region.
- Sweep key control factors (e.g., LoRA scale, strength, random seed) to increase diversity.
- Apply YOLO-based posterior filtering to reject low-consistency samples.
- Export accepted images into a structured output folder for downstream training.

---

#### 2.2 GPPL (Cavity Expansion)
- Segment steel anchor and cavity regions to obtain masks.
- Compute PCA principal axis of the steel anchor region.
- Enforce half-space constraint to restrict physically plausible growth direction.
- Apply stepwise axial expansion with a defined step length and number of steps.
- Align gray-scale statistics via photometric alignment (e.g., median-based).
- Apply boundary feather blending to smooth transitions and reduce artifacts.
- Export intermediate visualizations (optional) and final augmented images.

---

#### 2.3 Boundary Continuity Evaluation (ΔG / ΔC)
- Compute boundary gradient difference ΔG within a narrow band around defect boundaries.
- Compute local contrast difference ΔC within the same band.
- Perform clean vs augmented comparison using matched boundary locations.
- Export per-sample results into a CSV for statistical testing and plotting.

---

#### 2.4 YOLO11m-OBB Full Chain (Training + GUI)
- Train a YOLO11m-OBB detector on clean-only vs augmented datasets to quantify gains.
- Evaluate standard detection metrics (mAP, IoU-related measures) and compare against baselines.
- Use the Qt GUI tool for inference visualization, qualitative inspection, and grading integration (if enabled).

---

## 🧠 Models and Resources

The trained weights used in our experiments are stored in the following directories (relative to the repository root):

- **LoRA weights (SLMY / DreamBooth-LoRA checkpoint):**  
  `SLMY/lora-ssfracture/checkpoint-500/`

- **YOLO detection weights for posterior filtering (SLMY):**  
  `SLMY/YOLO-Det_filter_weights/`

- **YOLO segmentation weights (GPPL / MODEL_WTS):**  
  `GPPL/MODEL_WTS/`

- **YOLO11m-OBB downstream deployment weights and related resources:**  
  `Engineering_Deployment/`

> If you cannot redistribute weights, please document the acquisition method (official release link / training procedure) in a private note or in the repository wiki.

---

## 📎 Supplementary Materials

Additional materials are hosted at:
👉 https://github.com/z0810gc/SLGP-DA

Includes:
1. Representative qualitative results and paper figures used in this repository (`RE_IMAGE/`, e.g., Fig. 5/10/11/12/14/15)
2. Core implementation scripts for the full SLGP-DA pipeline (SLMY / GPPL / ΔGΔC evaluation / YOLO11m-OBB GUI)
3. Example configuration conventions and folder templates for reproducibility (data organization, naming alignment, and weight placement)

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

Feel free to cite or reuse this project for academic and research purposes.  
Issues and contributions are welcome.
