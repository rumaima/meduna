# MedUnA 🩺  
🏥 Accepted at ISBI 2025
**Language-Guided Unsupervised Adaptation of Vision-Language Models for Medical Image Classification**

> Can Language-Guided Unsupervised Adaptation Improve Medical Image Classification Using Unpaired Images and Texts?  
> **Umaima Rahman**, Raza Imam, Mohammad Yaqub, Boulbaba Ben Amor, Dwarikanath Mahapatra  
> **Affiliations**: MBZUAI · Inception · Monash University  
> [[📄 Paper]](https://arxiv.org/abs/2409.02729)

---

## 🔍 Overview

Medical image classification is often hindered by the scarcity of expert-labeled data. MedUnA addresses this limitation by **leveraging unpaired images and large language model (LLM)-generated class descriptions**, enabling **unsupervised adaptation** of a vision-language model (VLM) without requiring paired image-text data or labels.

We attach a lightweight **cross-modal adapter** to the visual encoder of MedCLIP and use a **contrastive entropy loss** to align visual and textual embeddings, improving classification performance in label-scarce settings.

---

## ✨ Key Highlights

- ✅ **Label-free tuning (LFT)** using unpaired images and text
- 🔁 Contrastive training with entropy loss and prompt tuning
- 📚 Uses rich LLM-generated descriptions per class
- 🧪 Evaluated on 5 diverse medical image datasets
- 🔌 No pretraining on image-text pairs required

---

## 📊 Datasets & Results

We evaluate MedUnA on:
- **Chest X-ray** datasets: Shenzhen TB, Montgomery TB, Guangzhou Pneumonia
- **Multi-class** datasets: IDRID, Skin Lesions (ISIC 2018)

MedUnA consistently improves accuracy over zero-shot VLM baselines, demonstrating strong out-of-the-box generalization.

---

## ⚙️ Setup

### 1️⃣ Clone the repo and create environment

```bash
git clone https://github.com/rumaima/meduna.git
cd meduna
conda create -n meduna_env python=3.9
conda activate meduna_env
pip install -r meduna_requirements.txt
```

---

## 2️⃣ 📦 Dataset Registration (For New Datasets)

If you're using a **new dataset** that is not already supported, you must **register it first**:

* Add your dataset config `.yaml` to the `configs/` folder
* Define the following in your config:

  * `dataset_path`
  * `class_names`
  * `image_loader` format (e.g., grayscale, RGB, resize, transforms)
* You may also need to update dataset-specific logic in:

  * `data/`
  * `scripts/`

---

## 3️⃣ 🚀 Run Training + Evaluation

Once your dataset is registered, run the following script:

```bash
bash scripts/LaFTer.sh <dataset_name>
```

Replace `<dataset_name>` with one of the following:

* `montgomery_cxr`
* `shenzhen_cxr`
* `guangzhou_pneumonia`
* `idrid`
* `isic2018`
* *or your custom dataset name*

**Example:**

```bash
bash scripts/LaFTer.sh idrid
```

This will:

* Load the corresponding YAML config
* Train using **MedUnA** (unsupervised, label-free)
* Save checkpoints and evaluation metrics

---

## 🧠 Method Summary

![MedUnA Framework](assets/meduna_framework.png)

* 📝 **Text encoder** processes LLM-generated class descriptions into embeddings
* 🖼️ **Visual encoder** (from MedCLIP) processes medical images
* 🧹 A lightweight **adapter** aligns visual embeddings to the text space
* 🧲 **Contrastive entropy loss** encourages better class separation and modality alignment
* ❌ Training is done **without any labels or paired supervision**

---

## 📁 Repository Structure

```
meduna/
├── configs/            # YAML configs for datasets
├── scripts/            # Training/evaluation bash scripts
├── models/             # Model components (adapter, projector)
├── data/               # Dataset loaders
├── utils/              # Helper functions
├── assets/             # Diagrams & figures
├── README.md           # You're here!
```

---

## 📚 Citation
Please cite our work if you find it useful :)

```bibtex
@misc{rahman2024meduna,
  title={Can Language-Guided Unsupervised Adaptation Improve Medical Image Classification Using Unpaired Images and Texts?},
  author={Umaima Rahman and Raza Imam and Mohammad Yaqub and Boulbaba Ben Amor and Dwarikanath Mahapatra},
  year={2024},
  url={https://github.com/rumaima/meduna}
}
```

---


