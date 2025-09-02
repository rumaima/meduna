# MedUnA 🩺  
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

Dataset Registration (For New Datasets)

If you're using a new dataset that is not already supported, you must register it first:

Add your dataset config YAML to the configs/ folder

Define:

dataset_path

class_names

image_loader format

You may need to update dataset-specific logic in data/ or scripts/
