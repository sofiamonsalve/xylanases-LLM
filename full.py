# all the packages are already in  our conda

from huggingface_hub import notebook_login

# paste your huggingface token 
#TODO
#notebook_login()

# select model
model_checkpoint = "facebook/esm2_t12_35M_UR50D"

# These are the URLs used for the download of the local files
# command for the download: 
'''
xylanases_fasta_url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28%28ec%3A3.2.1.8%29+OR+%28ec%3A3.2.1.32%29+OR+%28ec%3A3.2.1.136%29+OR+%28ec%3A3.2.1.156%29+AND+%28cc_catalytic_activity_exp%3A*%29%29"
xylanases_tsv_url = "https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Corganism_name%2Ccc_catalytic_activity%2Cec%2Corganism_id&format=tsv&query=%28%28ec%3A3.2.1.8%29+OR+%28ec%3A3.2.1.32%29+OR+%28ec%3A3.2.1.136%29+OR+%28ec%3A3.2.1.156%29+AND+%28cc_catalytic_activity_exp%3A*%29%29"

enzymes_not_x_fasta_url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28NOT+%28ec%3A*%29%29+AND+%28reviewed%3Atrue%29"
enzymes_not_x_tsv_url = "https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Corganism_name%2Ccc_catalytic_activity%2Cec%2Corganism_id&format=tsv&query=%28NOT+%28ec%3A*%29%29+AND+%28reviewed%3Atrue%29"

not_enzyme_fasta_url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28NOT+%28ec%3A*%29%29+AND+%28reviewed%3Atrue%29"
not_enzyme_tsv_url = "https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Corganism_name%2Ccc_catalytic_activity%2Cec%2Corganism_id&format=tsv&query=%28NOT+%28ec%3A*%29%29+AND+%28reviewed%3Atrue%29"

mmseq_fasta_url = "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/4c63552462781a83b9b897714c076adc93ab050e?format=fasta"
mmseq_tsv_url = "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/4c63552462781a83b9b897714c076adc93ab050e?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Corganism_name%2Ccc_catalytic_activity%2Cec&format=tsv"
'''

# fetch fasta and return dictionary for URLs
'''
def fetch_fasta(url):
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    fasta_data = response.text.split('\n')
    sequences = {}
    current_key = None
    current_seq = []
    for line in fasta_data:
        if line.startswith(">"):
            if current_key is not None:
                sequences[current_key] = "".join(current_seq)
                current_seq = []
            current_key = line[1:].split()[0]  # Use the ID as the key, excluding the ">"
        else:
            current_seq.append(line.strip())
    if current_key is not None:
        sequences[current_key] = "".join(current_seq)
    return sequences
'''
# function to load the fasta file 
def load_fasta_from_file(file_path):
    with open(file_path, 'r') as file:
        fasta_data = file.readlines()

    sequences = {}
    current_key = None
    current_seq = []

    for line in fasta_data:
        line = line.strip()  # Remove any leading/trailing whitespace
        if line.startswith(">"):
            if current_key is not None:
                sequences[current_key] = "".join(current_seq)
                current_seq = []
            current_key = line[1:].split()[0]  # Use the ID as the key, excluding the ">"
        else:
            current_seq.append(line)

    if current_key is not None:
        sequences[current_key] = "".join(current_seq)

    return sequences

# loading the tsv file

import pandas as pd

# load the fasta files
xylanases_fasta = load_fasta_from_file('local_data/xylanases.fasta')
enzymes_not_x_fasta = load_fasta_from_file('local_data/enzymes_not_x.fasta')
#not_enzyme_fasta = load_fasta_from_file('local_data/not_enzyme.fasta')
mmseq_fasta = load_fasta_from_file('local_data/mmseq_fasta.fasta')

# load the tsv files 

# xylanase
df_xylanases = pd.read_csv('local_data/xylanases.tsv', sep='\t')  

# enzymes not xylanases
df_enzymes_not_x = pd.read_csv('local_data/enzymes_not_x.tsv', sep='\t')

# not enzyme
#df_not_enzyme = pd.read_csv('local_data/not_enzyme.tsv', sep='\t')

# mmseq 
df_mmseq = pd.read_csv('local_data/mmseq_tsv.tsv', sep='\t') 

print(df_xylanases.head(), df_enzymes_not_x.head(), df_mmseq.head())


# clean fasta dictionary keys
def clean_dict_keys(original_dict):
    for key in list(original_dict.keys()):
        if '|' in key:
            new_key = key.split('|')[1]
            # Move value to new key and remove old key-value pair
            original_dict[new_key] = original_dict.pop(key)
    return original_dict

# run code for all dictionaries
xylanases_fasta = clean_dict_keys(xylanases_fasta)
enzymes_not_x_fasta = clean_dict_keys(enzymes_not_x_fasta)
#not_enzyme_fasta = clean_dict_keys(not_enzyme_fasta)
mmseq_fasta_fasta = clean_dict_keys(mmseq_fasta)

# Get 'reviewed' entries from df_xylanase
reviewed_xylanase_ids = df_xylanases.loc[df_xylanases['Reviewed'] == 'reviewed', 'Entry'].tolist()

# Get sequences for 'reviewed' xylanase entries from xylanases_fasta
xylanase_test_data = [
    (entry_id, xylanases_fasta[entry_id], 'df_xylanases',
     df_xylanases.loc[df_xylanases['Entry'] == entry_id, 'EC number'].iloc[0], 1)
    for entry_id in reviewed_xylanase_ids
]


# for training/test: use 50% xylanase and 50% non-xylanase/non-enzyme
# determine number of negative examples necessary
num_negative_samples_each = len(reviewed_xylanase_ids) // 2

#TODO
# Select half of the xylanase entries from mmseq_ids and df_enzymes_not_x
mmseq_ids = df_mmseq['Entry'].sample(n=num_negative_samples_each).tolist()
enzymes_not_x_ids = df_enzymes_not_x['Entry'].sample(n=num_negative_samples_each).tolist()

# Get sequences
mmseq_test_data = [
    (entry_id, mmseq_fasta[entry_id], 'df_mmseq',
     df_mmseq.loc[df_mmseq['Entry'] == entry_id, 'EC number'].iloc[0], 0)
    for entry_id in mmseq_ids
]

enzymes_not_x_test_data = [
    (entry_id, enzymes_not_x_fasta[entry_id], 'df_enzymes_not_x',
     df_enzymes_not_x.loc[df_enzymes_not_x['Entry'] == entry_id, 'EC number'].iloc[0], 0)
    for entry_id in enzymes_not_x_ids
]

# Combine all data into a single test set
test_data = xylanase_test_data + mmseq_test_data + enzymes_not_x_test_data
df_test = pd.DataFrame(test_data, columns=['id', 'sequence', 'source', 'EC number','target'])

print(df_test.head(), df_test.shape)
target_counts = df_test['target'].value_counts()
print(f"number of xylanases (1) {target_counts.get(1, 0)}, and not xylanases (0) {target_counts.get(0, 0)}")

# Training data

# similar steps as above to create a training dataset

# Select roughly 1214 unreviewed entries from df_xylanase
unreviewed_xylanase_ids = df_xylanases.loc[df_xylanases['Reviewed'] != 'reviewed', 'Entry'].sample(n=714).tolist()
# Get sequences for these entries
xylanase_train_data = [(entry_id, xylanases_fasta[entry_id], 'df_xylanase', df_xylanases.loc[df_xylanases['Entry'] == entry_id, 'EC number'].iloc[0], 1) for entry_id in unreviewed_xylanase_ids]

# again 50% xylanase examples, and 50% non-xylanase/mmseq examples
num_entries_other = (1428 - len(xylanase_train_data)) // 2
enzymes_not_x_ids = df_enzymes_not_x['Entry'].sample(n=num_entries_other).tolist()
mmseq_ids = df_mmseq['Entry'].sample(n=num_entries_other).tolist()

# Get sequences
enzymes_not_x_train_data = [(entry_id, enzymes_not_x_fasta[entry_id], 'df_enzymes_not_x', df_enzymes_not_x.loc[df_enzymes_not_x['Entry'] == entry_id, 'EC number'].iloc[0], 0) for entry_id in enzymes_not_x_ids]
mmseq_train_data = [(entry_id, enzymes_not_x_fasta[entry_id], 'df_mmseq', df_mmseq.loc[df_mmseq['Entry'] == entry_id, 'EC number'].iloc[0], 0) for entry_id in mmseq_ids]

# Combine all data into a single training set
train_data = xylanase_train_data + enzymes_not_x_train_data + mmseq_train_data
df_training = pd.DataFrame(train_data, columns=['id', 'sequence', 'source', 'EC number', 'target'])

print(df_training.head(), df_training.shape)
target_counts = df_training['target'].value_counts()
print(f"number of xylanases (1) {target_counts.get(1, 0)}, and not xylanases (0) {target_counts.get(0, 0)}")

# let's check if there is overlap between test and training sets

test_ids = set(df_test['id'])
training_ids = set(df_training['id'])
# Find the common ids
common_ids = test_ids.intersection(training_ids)
if common_ids:
    print(f"There are {len(common_ids)} overlapping ids between test and training sets: {common_ids}")
else:
    print("No overlapping ids between test and training sets.")


# Keep only entries where the sequence length is <= 1024n (probably better to take 1000))
df_training = df_training.loc[df_training['sequence'].apply(len) <= 1024]
df_test = df_test.loc[df_test['sequence'].apply(len) <= 1024]

# prepare data for LLM training
train_sequences = df_training['sequence'].tolist()
test_sequences = df_test['sequence'].tolist()

# Tokenising

from transformers import AutoTokenizer

# SPECIFIC FOR YOUR EARLIER SELECTED MODEL
# model_checkpoint = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# PREPARE TOKENIZED DATA - USED FOR LLM FINE-TUNING/PREDICTION
train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)

# Dataset creation

from datasets import Dataset

# Extract labels
# 'target' is the df column for our prediction target
train_labels = df_training['target'].tolist()
test_labels = df_test['target'].tolist()
# Convert tokenized data to datasets
train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)
# Add our labels (for classification)
train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)

# Model loading and training

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = max(train_labels + test_labels) + 1  # 0 can be a label! (so add 1)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

from transformers import TrainingArguments

model_name = model_checkpoint.split("/")[-1]
batch_size = 8

# HYPERPARAMETERS AFFECT PERFORMANCE and ACCURACY
args = TrainingArguments(
    f"{model_name}-ft-xylanase",  # NAME OF THE MODEL SAVED TO HUGGINFACE, CHANGE IF NEEDED
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)
# the first argument is the name that will be used to save the final trained model to huggingface
# you can always reload the model later using AutoModelForSequenceClassification.from_pretrained("your_model_name")

from evaluate import load
import numpy as np

# use accuracy as metric
# SENSITIVE TO IMBALANCED DATASETS!!!
metric = load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

print("Happy times")