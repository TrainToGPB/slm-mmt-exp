# built-in
import os
import sys
import random

# third-party
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import wandb
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from nltk.translate.bleu_score import corpus_bleu

# custom
sys.path.append('../../../../')
from custom_utils.training_utils import set_seed


def load_model_and_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang='en_XX', tgt_lang='ko_KR')
    return model, tokenizer


def preprocess_example(example, tokenizer):
    src_text = example['en']
    tgt_text = example['ko']

    src_encoding = tokenizer.encode_plus(src_text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    tgt_encoding = tokenizer.encode_plus(tgt_text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

    return {
        'input_ids': src_encoding['input_ids'].squeeze(), 
        'label_ids': tgt_encoding['input_ids'].squeeze()
    }


def compute_metrics(pred):
    pred_ids = pred.predictions[0]
    label_ids = pred.label_ids

    bleu_score = corpus_bleu([[label] for label in label_ids], pred_ids)

    return {"BLEU": bleu_score}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def train(SEED, OUTPUT_DIR, PLM_NAME, DATASET_NAME, PROJECT_NAME, RUN_NAME):
    set_seed(SEED)

    model, tokenizer = load_model_and_tokenizer(PLM_NAME)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    dataset = load_dataset(DATASET_NAME)
    dataset = dataset.map(lambda example: preprocess_example(example, tokenizer))

    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    num_train_epochs = 5
    dataset_size = len(dataset['train'])
    per_device_train_batch_size = 8
    per_device_eval_batch_size = per_device_train_batch_size
    warmup_steps = int(num_train_epochs * (dataset_size / per_device_train_batch_size) * 0.1)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        dataloader_num_workers=4,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        learning_rate=3e-5,
        warmup_steps=warmup_steps,
        lr_scheduler_type='linear',
        gradient_checkpointing=True,
        # gradient_accumulation_steps=4,
        weight_decay=0.01,
        fp16=True,
        logging_dir='./logs',
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        seed=SEED,
        report_to='wandb',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == '__main__':
    SEED = 42
    OUTPUT_DIR = '../models'
    PLM_NAME = 'google/mt5-base'
    DATASET_NAME = 'traintogpb/aihub-koen-translation-integrated-tiny-100k'
    PROJECT_NAME = 'ojt_translation'
    # PROJECT_NAME = 'huggingface'
    RUN_NAME = 'mt5-test'

    train(SEED, OUTPUT_DIR, PLM_NAME, DATASET_NAME, PROJECT_NAME, RUN_NAME)
