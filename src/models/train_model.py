# Necessary inputs
import warnings

from datasets import load_dataset, load_metric
import transformers
import datasets
import random
import pandas as pd
import numpy as np
import torch
from IPython.display import display, HTML

warnings.filterwarnings('ignore')

from transformers import AutoModel, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig, BartForConditionalGeneration, AutoTokenizer

base_model_name = 'facebook/bart-base'
model_name = 'SkolkovoInstitute/bart-base-detox'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# setting random seed for transformers library
transformers.set_seed(42)

# Load the BLUE metric
metric = load_metric("sacrebleu")

df = pd.read_csv("../../data/interim/processed.tsv", sep ="\t")
df = df.rename(columns={"translation": "target", "reference": "text"})

train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
train_dataset = datasets.Dataset.from_dict(train)
test_dataset = datasets.Dataset.from_dict(test)
validation_dataset = datasets.Dataset.from_dict(validate)
raw_datasets = datasets.DatasetDict({"train":train_dataset,"test":test_dataset, "validation": validation_dataset})

# prefix for model input
prefix = "Detoxify "

max_input_length = 128
max_target_length = 128
source_lang = "toxic"
target_lang = "detoxified"
padding = "max_length"


def preprocess_function(examples):
    """
    This function tokenizes all words in a given batch

    Args:
        examples: a batch of n samples from the raw dataset
    """
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["target"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    
    
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# defining the parameters for training
batch_size = 32
model_name = base_model_name.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-detoxify",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    report_to='tensorboard',
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# simple postprocessing for text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# compute metrics function to pass to trainer
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
    
    
# instead of writing train loop we will use Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# saving model
trainer.save_model('best')

