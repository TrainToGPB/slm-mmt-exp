"""
Train mBART with QLoRA training

The following functions are available:
- load_model_and_tokenizer: Load model and tokenizer
- preprocess_example: Preprocess example
- postprocess_text: Postprocess text
- preprocess_logits_for_metrics: Preprocess logits for metrics
- calculate_warmup_steps: Calculate warmup steps
- train: Train model

Example:
    $ python train_mbart_qlora.py

Notes:
- The training arguments are defined in the mbart_qlora_config.yaml file.
"""
# built-in
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys

# third-party
import torch
import numpy as np
import wandb
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from peft import LoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# custom
sys.path.append('../../../../')
from custom_utils.training_utils import set_seed
from custom_utils.general_secret import WANDB_CLIENT_KEY
from custom_utils.argument import parse_arguments_mbart


def load_model_and_tokenizer(plm_name, bnb_config, lora_config, src_lang='en_XX', tgt_lang='ko_KR'):
    """
    Load model and tokenizer

    Args:
    - plm_name (str): Pretrained model name
    - bnb_config (BitsAndBytesConfig): BitsAndBytesConfig
    - lora_config (LoraConfig): LoraConfig
    - src_lang (str): Source language
    - tgt_lang (str): Target language

    Returns:
    - model (PreTrainedModel): Pretrained model
    - tokenizer (PreTrainedTokenizer): Pretrained tokenizer
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        plm_name,
        quantization_config=bnb_config,
        torch_dtype=bnb_config.bnb_4bit_compute_dtype,
        device_map=torch.cuda.current_device(),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        plm_name, 
        src_lang=src_lang, 
        tgt_lang=tgt_lang,
        trust_remote_code=True,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def preprocess_example(example, max_length, tokenizer):
    """
    Preprocess example

    Args:
    - example (dict): Example dictionary
    - max_length (int): Maximum length of input sequence
    - tokenizer (PreTrainedTokenizer): Pretrained tokenizer

    Returns:
    - model_inputs (dict): Model inputs
    """
    src_text = example['en']
    tgt_text = example['ko']

    model_inputs = tokenizer(
        src_text, 
        text_target=tgt_text, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True,
        return_tensors="pt"
    )
    model_inputs = {key: value.squeeze() for key, value in model_inputs.items()}

    return model_inputs


def postprocess_text(preds, labels):
    """
    Postprocess text

    Args:
    - preds (list): Predictions
    - labels (list): Labels

    Returns:
    - preds (list): Postprocessed predictions
    - labels (list): Postprocessed labels
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
    """
    Calculate warmup steps

    Args:
    - epochs (int): Number of epochs
    - dataset_size (int): Size of dataset
    - batch_size (int): Batch size
    - gradient_accumulation_steps (int): Gradient accumulation steps
    - warmup_ratio (float): Warmup ratio

    Returns:
    - warmup_steps (int): Warmup steps
    """
    steps_per_epoch = (dataset_size / batch_size)
    total_steps = epochs * steps_per_epoch / gradient_accumulation_steps
    total_steps_per_device = total_steps / torch.cuda.device_count()
    warmup_steps = int(total_steps_per_device * warmup_ratio)
    return warmup_steps


def train(args):
    """
    Train model

    Args:
    - args (argparse.ArgumentParser): Input arguments
    """
    set_seed(args.seed)

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_double_quant
    )
    modules = args.lora_target_modules
    print("LoRA adapted modules:", modules)
    lora_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias='none',
        target_modules=modules,
        task_type='SEQ_2_SEQ_LM'
    )

    model, tokenizer = load_model_and_tokenizer(
        args.plm_name, 
        bnb_config,
        lora_config,
        args.src_lang, 
        args.tgt_lang
    )

    dataset = load_dataset(args.dataset_name)
    dataset = dataset.map(lambda example: preprocess_example(example, args.max_length, tokenizer))

    metric = evaluate.load('sacrebleu')

    dataset_size = len(dataset['train'])
    warmup_steps = calculate_warmup_steps(
        epochs=args.num_epochs,
        dataset_size=dataset_size,
        batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio
    )

    if args.just_test:
        project_name = 'test'
        output_dir = '../models/test'
        logging_steps = 1
        eval_steps = 1
        save_steps = 1
        train_dataset = dataset['train'].select(range(1000))
        eval_dataset = dataset['validation'].select(range(10))
    else:
        project_name = args.project_name
        output_dir = args.output_dir
        logging_steps = args.logging_steps
        eval_steps = warmup_steps
        save_steps = warmup_steps
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']

    wandb.login(
        anonymous='never',
        key=WANDB_CLIENT_KEY,
        relogin=True,
        force=True
    )
    wandb.init(project=project_name, name=args.run_name)

    warmup_steps = calculate_warmup_steps(
        epochs=args.num_epochs,
        dataset_size=len(dataset['train']),
        batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps, # 전체 step의 10%
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        logging_steps=logging_steps,
        eval_steps=eval_steps, # 전체 step의 10%
        save_steps=save_steps, # 전체 step의 10%
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
    )

    def compute_sacrebleu(p):
        """
        Compute SacreBLEU

        Args:
        - p (EvalPrediction): EvalPrediction object

        Returns:
        - result (dict): Result dictionary
        """
        preds, labels = p.predictions[0], p.label_ids
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        for decoded_label, decoded_pred in zip(decoded_labels[:10], decoded_preds[:10]):
            print("[LABEL]")
            print(decoded_label[0])
            print("[PREDICTION]")
            print(decoded_pred)

        result = metric.compute(references=decoded_labels, predictions=decoded_preds)
        result = {"SacreBLEU": result["score"]}
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_sacrebleu,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    yaml_path = './mbart_qlora_config.yaml'
    args = parse_arguments_mbart(yaml_path)
    train(args)
