# 🇰🇭 Khmer Text Style Transformer (khmer-tst)

<div align="center">

**Bridging the gap between Common Khmer and Royal Registers (Reacheasap)**

</div>



## 📖 Overview

The **Khmer Text Style Transformer** is an NLP project dedicated to the preservation and digital transformation of the Khmer language's unique sociolinguistic registers. Specifically, it focuses on **Reacheasap (រាជាសព្ទ)** — the highly formalized royal register used when referring to or addressing the Monarchy, Clergy, and in classical literature.

### 🎯 Project Objectives

The project automates the transition between everyday vernacular and the sophisticated, honorific-heavy structures of formal Khmer.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 
    'primaryColor': '#f0f0f0', 
    'primaryBorderColor': '#000000', 
    'lineColor': '#1f77b4',   %% arrows in blue
    'textColor': '#000000',
    'edgeLabelBackground':'#f0f0f0'
}}}%%
graph LR
    A[Data Acquisition] --> B[Corpus Cleaning]
    B --> C[ML Pre-training]
    C --> D[Domain Fine-tuning]

```




---

## ✨ Key Features

* **Register Transformation:** Seamlessly convert text between **Common (ធម្មតា)** and **Royal (រាជាសព្ទ)** registers.
* **Curated Parallel Corpus:** A high-quality dataset of 690+ manually verified Khmer sentence pairs.
* **Automated Scrapers:** Custom scripts designed to harvest rich Khmer linguistic data from Wikipedia and traditional folklore sources.
* **End-to-End Pipeline:** Includes notebooks for everything from data scraping and Masked Language Modeling (MLM) to final fine-tuning.

---

## 📊 Linguistic Dataset

The core of this project is a parallel corpus of **690+ verified pairs**, capturing the nuance of Khmer honorifics.

| Register   | Example Sentence                                    |
| ---------- | --------------------------------------------------- |
| **Common** | គាត់ឆ្លងកាត់ឆ្នេរសមុទ្រ បានទៅដល់កោះមួយ              |
| **Royal**  | ព្រះអង្គនិមន្ដកាត់តាមឆ្នេរមហាសាគរ បានយាងទៅដល់កោះមួយ |

### Transformation Logic Breakdown

| Word (Common)       | Word (Royal)     | English Equivalent    |
| ------------------- | ---------------- | --------------------- |
| គាត់ (He/She)       | **ព្រះអង្គ**     | His/Her Majesty       |
| ញ៉ាំ (Eat)          | **សោយ**          | To partake (Royal)    |
| មក (Come)           | **យាងមក**        | To proceed / arrive   |
| ឆ្នេរសមុទ្រ (Coast) | **ឆ្នេរមហាសាគរ** | The Great Ocean Coast |

---

## 📁 Project Structure

```bash
khmer-tst/
├── 📊 data/
│   ├── general-text.txt       # Unstructured Khmer corpus for pre-training
│   └── normal-royal.csv       # The parallel register dataset
├── 🤖 models/                  # Saved weights and transformer checkpoints
├── 📓 notebooks/
│   ├── 01-scraping.ipynb      # Wikipedia & Folklore data collection
│   ├── 02-pre-training.ipynb  # MLM on general Khmer text
│   └── 03-fine-tuning.ipynb   # Sequence-to-sequence register training
└── README.md
```

---

## 🧠 Model Training Workflow

The project follows a **two-phase training strategy** to develop a robust Khmer Text Style Transformer. The architecture is based on an **LSTM encoder–decoder framework**, trained progressively to first learn general language structure, then adapt to register-specific nuances.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 
    'primaryColor': '#f0f0f0', 
    'primaryBorderColor': '#000000', 
    'lineColor': '#1f77b4',   %% arrows in blue
    'textColor': '#000000',
    'edgeLabelBackground':'#f0f0f0'
}}}%%
graph TD
    A[Unsupervised Pre-Training] --> B[Fine-Tuning on Parallel Corpus]
    B --> C[Inference & Style Transformation]
```


### **1. Pre-Training Phase**

During this stage, the model is exposed to a **large unpaired Khmer text corpus** to learn the foundational syntax, morphology, and semantics of the language.

**Objective:**
Train the LSTM to predict the next token in a sequence, allowing it to learn contextual representations of Khmer words.

**Setup:**

* **Dataset:** `general-text.txt` (unstructured Khmer corpus)
* **Task:** Next-word prediction (language modeling)
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam
* **Learning Rate:** 0.001
* **Epochs:** 20–30 (with early stopping)

This phase enables the model to build **general language representations**, capturing grammar and sequential dependencies in Khmer text.

---

### **2. Fine-Tuning Phase**

After pre-training, the model is fine-tuned on a **parallel corpus of Common–Royal sentence pairs**. This teaches the model to transform sentences from the **Common register** to the **Royal register**, leveraging the linguistic foundation learned during pre-training.

**Objective:**
Adapt the pretrained LSTM for **sequence-to-sequence translation** between registers.

**Setup:**

* **Dataset:** `normal-royal.csv` (690+ aligned sentence pairs)
* **Task:** Text style transfer (Common → Royal)
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam (lower learning rate for stability)
* **Evaluation Metrics:** BLEU score, perplexity
* **Model Checkpointing:** Best model saved as `best_model.pt`

This phase refines the model to capture **register-specific lexical and stylistic transformations**, producing fluent and contextually accurate royal text.

---

## 🛠️ Tech Stack

* **Linguistic Processing:** Custom tokenization for Khmer script.
* **Web Tools:** `BeautifulSoup4`for scraping traditional literature.
* **AI/ML:** `Pandas`, `PyTorch`, and `NumPy` for data processing and LSTM implementation.

---



<div align="center">

**Preserving Khmer Heritage through Modern Technology**

If this project helps your research, please consider giving it a ⭐!

</div>

---

