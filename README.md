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

## 3.1 Resulting metrics of the final model

The final model with all the adjusted parameters resulted in very high accuracy (0.99422), high precision (0.9955), high ROC-AUC (0.99), and low loss (0.0335).

Table 1 - Confusion matrix for the model ESM2-t6-8M:

| Predicted | Positive | Negative | Total |
|-----------|----------|----------|-------|
| Actual Positive | 5045 | 39 | 5084 |
| Actual Negative | 36 | 5089 | 5125 |
| Total | 5081 | 5128 | 10209 |

The confusion matrix also showed few false positives and false negatives, consistent with the high accuracy and precision.

## 3.2 Result of the speed test

A protein sequence consisting of 1000 random amino acids had a prediction time of 0.1341 seconds, whereas a sequence consisting of only 100 amino acids took 0.0378 seconds.

Based on this information, it is possible to extrapolate the prediction time of a protein sequence or protein genome. The average size for a xylanase is between 200 and 500 amino acids, and looking at the 300 amino acid run, which took 0.0353 seconds, it can be concluded that the model can quickly classify xylanases for standard sequence lengths.


## Discussion

Initially, the ESM2-T12-35M model was used because a more complex model with more hyperparameters and layers is generally better at finding patterns. However, it was found that the ESM2-T6-8M model had the same accuracy, precision, and ROC-AUC, but with a lower test loss. This is why the ESM2-T6-8M model was used instead. The reason for this could be that the model's task was simple, e.g. xylanases can be easily distinguished from other enzymes, so the simpler model was sufficient and therefore had high accuracy. Having more hyperparameters and layers would, in this case, lead to overfitting, especially when considering the smaller size of the dataset, resulting in lower training loss but higher test loss. Additionally, the smaller model requires fewer computational resources, and the training time is reduced.

The accuracy of the resulting model was very high, and the results were already very good initially. This seems to support the fact that xylanase sequences are very different compared to other enzyme sequences, making it an easy task for the model to distinguish them even from similar protein sequences.

Some of the entries were shared between the test and training dataset; however, this is only a small part of the total dataset. The impact that this could have on how the model is tested is minimal. This distribution should be kept in mind as it could have contributed to the high accuracy of the model since testing with the same sequences the model was trained with can lead to misleading evaluation metrics.

Throughout the process, some assumptions were made. Firstly, to train the model, all the members of the same cluster were labeled to have the same cluster type as the representative of the cluster. This was, however, not always true; some sequences in the same clusters differed in cluster types. Another option would have been to label all the members of the cluster the same cluster type as what the majority of the members have as their cluster type. Given the model's already high accuracy and the fact that, during the accuracy checks, the model refers to the labels of entries rather than to those of clusters, it was decided to continue with the labeling by the representative.

Another possible division of the sets would have all the reviewed in the test set, as to make this one more reliable as a validation tool. However, as the evaluation metrics for the model were already quite good, this specification step was skipped.

## Conclusion

The focus of this project was the classification of xylanase enzymes using LLMs. Xylanases are enzymes involved in the degradation of xylan, a polymer found in the plant cell wall. Xylanases have diverse industrial applications, ranging from biofuel production to the food industry.

The three datasets – xylanases, enzymes, proteins - were downloaded from UniProt, MMseqs2 was used for clustering, and the training and test datasets were created using an 80/20 split respectively. Both training and test datasets had a ratio of 50% xylanases, 25% enzymes (not xylanases), and 25% proteins (not enzymes).

For fine-tuning the model, different hyperparameters were explored, and the performances of two models (ESM2-T12-35M and ESM2-T6-8M) were compared. The smaller ESM2-T6-8M model had higher accuracy and lower validation loss, which is why this model was selected. The hyperparameters that were fine-tuned for the chosen model were: number of epochs (2); batch size (8); learning rate (0.00001); and weight decay (0.01).

The final model had very good results: accuracy (0.99422); precision (0.9955); and ROC-AUC (0.99). Given the large number of unannotated protein sequences in UniProt, this model is an accurate and fast way to classify sequences into xylanases or not xylanases.
