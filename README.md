# Deep Learning Based fMRI Denoising: A Comparative Analysis of 3D U-Net Attention and Wasserstein GAN Architectures for fMRI Denoising

This repository contains the official implementation for the Master's project "Deep Learning Based fMRI Denoising", comparing 3D U-Net Attention and 3DWGAN architectures. This study systematically develops and refines these models to enhance the signal-to-noise ratio (SNR) in functional Magnetic Resonance Imaging (fMRI) data.

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Iterative Model Comparison](#-iterative-model-comparison)
- [Qualitative Results](#-qualitative-results)
- [Repository Structure](#-repository-structure)
- [Setup and Usage](#-setup-and-usage)
- [Citation](#-citation)
- [License](#-license)

---

## 📝 Project Overview
Functional MRI (fMRI) data is inherently corrupted by significant noise, which compromises signal quality and limits neurobiological interpretations. This project tackles this challenge by implementing and comparing two state-of-the-art deep learning paradigms. The goal is to develop a robust model that can significantly reduce noise while preserving the fine-grained anatomical and structural details essential for accurate neuroscience research.

---

## 📊 Iterative Model Comparison

Our development process followed a methodical, iterative approach, primarily focused on the 3D U-Net architecture. Each attempt provided critical insights that informed the design of the next, culminating in a final optimized model. The table below summarizes this journey, highlighting the key decisions and their impact on performance.

| Feature / Metric | Model 1 (Baseline) | Model 2 (Data-Scaled) | Model 2 (Fine-Tuned Failure) | Model 3 (Final Optimized Model) |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Size** | 10 Files (~2.8k vol) | 20 Files (~5.6k vol) | Small Subsets (4-8 files) | **20 Files (~5.6k vol)** |
| **Architecture** | 3D Attn U-Net + **LSTM** | 3D Attn U-Net + **LSTM** | 3D Attn U-Net + **LSTM** | 3D Attn U-Net + **No LSTM** |
| **Loss Function** | `binary_crossentropy` | `binary_crossentropy` | `composite_loss` (α=0.8) | **`composite_loss` (α=0.5)** |
| **Noise Type/Level** | Rician Noise - Constant | Rician Noise - Constant | Rician Noise - Constant | **Rician Noise - Variable** |
| **Test PSNR (dB)** | 34.54 | 33.28 | 31.96 (Worst) | **37.62 (Highest)** |
| **Test SSIM** | 0.865 | 0.796 | 0.772 (Worst) | **0.9223 (Highest)** |

---

## ✨ Qualitative Results

The final optimized model (Model 3) demonstrates a remarkable ability to restore anatomical structures from heavily corrupted inputs.

| Noisy Input | Denoised Output (Model 3) | Ground Truth |
| :---: | :---: | :---: |
| ![Noisy Input](result/3d_unet_attention_model3_output.png) | ![Denoised Output](result/3d_unet_attention_model3_best.png) | *(Buraya orijinal görüntünün olduğu bir resim eklenebilir)* |

*Note: Visual outputs for all models can be found in the `/result` directory.*

---

## 📂 Repository Structure

```
.
├── 3D_UNet_Attention_Architecture/
│   ├── model_1/
│   ├── model_2/
│   └── model_3/
├── 3D_WGAN_Architecture/
├── result/
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Setup and Usage

### Prerequisites
A `requirements.txt` file is provided for easy setup. It is recommended to use a virtual environment.
```
tensorflow>=2.15.0
numpy
scipy
matplotlib
seaborn
ipywidgets
```

### Installation & Running
1.  Clone the repository and navigate into the directory.
2.  Install dependencies: `pip install -r requirements.txt`
3.  The Jupyter Notebooks (`.ipynb`) for each experiment are located in their respective architecture and model folders. It is recommended to run them in sequence to follow the project's progression.

---

## 📄 Citation

If you find this work useful for your research, please consider citing our project report:
```bibtex
@mastersthesis{Karabag2025FMRI,
  author  = {Gökhan Karabag and Mihir Joshi and Prajwal Shet and Aravind Gangavarapu and Shreyash Deokate},
  title   = {Deep Learning Based fMRI Denoising: 3D U-Net Attention, 3DWGAN},
  school  = {Technische Hochschule Ingolstadt},
  year    = {2025},
  month   = {June},
  note    = {Supervised by Prof. Dr. Marion Menzel \& Thomas Alan Loboy Ramos}
}
```

---

## ⚖️ License
This project is licensed under the MIT License.
