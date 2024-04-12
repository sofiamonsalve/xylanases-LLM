# Fine-tuning a Large Language Model for Xylanase Classification 

# Abstract

**Motivation:** Developing a Large Language Model (LLM) to classify xylanases addresses the challenge of limited manual annotation of protein sequences. This LLM offers a rapid and cost-effective solution for categorizing xylanases, facilitating a better understanding of their functions in biosciences.

**Results:** The fine-tuning of an ESM2-T6-8M model combined with MMseq2 clustering resulted in an effective strategy for accurately classifying xylanases based on shared characteristics.

## 2.1 Initial Datasets

Three datasets were utilized for creating the test and training sets:

1. **Xylanases Dataset:** Comprising reviewed sequences from Swiss-Prot and unreviewed sequences from TrEMBL, totaling 179 and 23,940 sequences respectively, encompassing EC numbers 3.2.1.8, 3.2.1.32, 3.2.1.136, and 3.2.1.156.

2. **Enzymes-not-Xylanases Dataset:** Consisting of sequences from UniProtKB with EC numbers starting with 3.2., excluding the four groups of xylanases. This dataset comprised 1,174 sequences, with 622 reviewed and 552 unreviewed entries.

3. **Non-Enzymes Dataset:** Comprising 295,259 reviewed sequences from UniProtKB, ensuring proteins lacked catalytic activity.

## 2.2 Hugging Face

Hugging Face, an online platform hosting machine learning models and datasets, was used to select the model. The chosen model, esm2_t12_35M_UR50D, is an ESM2 model with 12 layers and 35M parameters.

## 2.3 MMseqs2

MMseqs2, a tool for clustering and searching large sets of sequences based on similarity, was employed. Clustering was performed with 80% similarity to classify sequences into xylanase, non-xylanase, or non-enzyme clusters.

## 2.4 Training and Test Sets

A Python script was developed to generate training and test sets while maintaining a split of 50% xylanases, 25% enzymes-not-xylanases, and 25% non-enzymes. The datasets were split by clusters to ensure similar sequences were in the same set.

## 2.5 Fine-tuning and Model Evaluation

Four hyperparameters were manually studied for fine-tuning: number of epochs, weight decay, learning rate, and batch size. The performance of the model was evaluated using various metrics including validation loss, accuracy, precision, recall, ROC/AUC, and runtime.

## 2.6 Workflow and Scripts

Python and Bash were the primary languages used for workflow creation and model fine-tuning. JupyterLab, Visual Studio Code, and GitHub were utilized for interactive development, code editing, version control, and collaboration.

This comprehensive approach facilitated the development and evaluation of an efficient model for xylanase classification, with potential applications in biosciences.
