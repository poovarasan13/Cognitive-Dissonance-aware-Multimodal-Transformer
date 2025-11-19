
# CDMT: Cognitive Dissonance-aware Multimodal Transformer  
*A replicable reference implementation for multimodal meme emotion understanding and human–AI disagreement detection.*

---

## Overview

This repository provides the full PyTorch implementation of **CDMT**, a multimodal transformer designed to perform:

- Multimodal emotion classification (image + caption)
- Cognitive dissonance regularization for human-aligned affect
- Human–AI disagreement detection
- Cross-domain generalization on Hateful Memes
- Text-only emotion transfer via GoEmotions

The codebase includes all components needed for reproducibility: Full model architecture, training/evaluation pipelines, default hyperparameters, and instructions to regenerate results reported in the paper.

## Repository Structure
`cdmt/\  
│── cdmt_model.py\
│── cdmt_dataset.py\ 
│── train_cdmt.py \
│── eval_cdmt.py\
│── requirements.txt \`

## Installation

### 1. Clone the repository

git clone https://github.com/<your-username>/CDMT.git
cd CDMT

### 2. Create environment

`python3 -m venv cdmt-env source cdmt-env/bin/activate` 

### 3. Install dependencies

`pip install -r requirements.txt` 

Ensure PyTorch is installed with CUDA support.

## Datasets

CDMT supports three public datasets:

### **Memotion 2.0**

-   Images + captions + sentiment/emotion labels
    
-   Primary dataset for training and evaluation
    

### **Hateful Memes**

-   Multimodal hate classification dataset
    
-   Used for cross-domain transfer analysis
    

### **GoEmotions**

-   58K Reddit comments
    
-   Text-only dataset for initializing linguistic affect embeddings
    
All datasets are publicly available and require no private or proprietary data.

## Training

### Basic Training Command

`python train_cdmt.py \
    --train_csv data/memotion_train.csv \
    --val_csv data/memotion_val.csv \
    --image_root data/memotion_images/ \
    --epochs 15 \
    --batch_size 32` 

**Features enabled by default**

-   ViT-B/16 visual encoder
    
-   BERT-base text encoder
    
-   6-layer cross-modal fusion transformer
    
-   Cognitive Dissonance Regularization (CDR)
    
-   Margin-based manifold separation
    
-   Temperature-scaled softmax
    
-   Optional HAD head for disagreement detection

## Evaluation

To evaluate a saved checkpoint:

`python eval_cdmt.py \
    --checkpoint checkpoints/cdmt_best.pt \
    --test_csv memotion_test.csv \
    --image_root memotion_images/`

## Hardware Requirements

-   GPU: NVIDIA RTX 2080 / A100 recommended
    
-   VRAM: ≥ 12 GB
    
-   Training time: ~9–10 hours on Memotion 2.0 (FP16, batch=32)
