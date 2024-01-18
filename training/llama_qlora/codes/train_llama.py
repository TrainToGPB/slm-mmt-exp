# built-in
import os
import sys
import yaml
import argparse

# third-party
import torch
import numpy as np
import wandb
import evaluate
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from peft.tuners.lora import LoraLayer
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
import torch.distributed.checkpoint as DCP

# custom
sys.path.append('../../../../')
from custom_utils.training_utils import set_seed
from custom_utils.general_secret import WANDB_CLIENT_KEY
from custom_utils.argument import parse_arguments_llama


def load_model_and_tokenizer(
        plm_name,
        use_4bit,
        bnb_4bit_quant_type,
        bnb_4bit_compute_dtype,
        use_double_quant,
        device_map  
    ):
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant
    )

    model = AutoModelForCausalLM.from_pretrained(
        plm_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map=torch.cuda.current_device()
    )
    model.config.use_cache = False
    model.config.pretrining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(plm_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return model, tokenizer


def preprocess_example(example):
    en_sign = '<English>'
    ko_sign = '<한국어>'
    text = " ".join([en_sign, example['en'], ko_sign, example['ko']])
    return {'text': text}


def postprocess_text(preds, labels):
    """
    Postprocess the predicted and reference texts by stripping leading and trailing whitespaces.

    Parameters:
    - preds (list): List of predicted texts.
    - labels (list): List of reference texts.

    Returns:
    - tuple: Tuple containing postprocessed predicted and reference texts.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess logits and labels for metric computation by extracting predicted IDs.

    Parameters:
    - logits (tuple): Tuple containing model logits.
    - labels (tensor): Tensor containing label IDs.

    Returns:
    - tuple: Tuple containing predicted IDs and labels.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
    """
    Calculate the number of warm-up steps for a sequence-to-sequence model training.

    Parameters:
    - epochs (int): Number of training epochs.
    - dataset_size (int): Total size of the training dataset.
    - batch_size (int): Batch size per device.
    - gradient_accumulation_steps (int): Number of gradient accumulation steps.
    - warmup_ratio (float): Ratio of warm-up steps to total training steps.

    Returns:
    - int: Number of warm-up steps to be used during training.
    """
    steps_per_epoch = (dataset_size / batch_size)
    total_steps = epochs * steps_per_epoch / gradient_accumulation_steps
    total_steps_per_device = total_steps / torch.cuda.device_count()
    warmup_steps = int(total_steps_per_device * warmup_ratio)
    return warmup_steps


def merge_and_save_distcp(model, distcp_dir):
    state_dict = {
        "model": model.state_dict(),
    }
    DCP.load_state_dict(
        state_dict=state_dict,
        storage_reader=DCP.FileSystemReader(distcp_dir),
        no_dist=True,
    )
    model.load_state_dict(state_dict["model"])

    save_path = distcp_dir + '-merged.pth'
    torch.save({'model_state_dict': model.state_dict()}, save_path)


def train(args):
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICE'] = '2,3' # if args.use_fsdp else '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    model, tokenizer = load_model_and_tokenizer(
        args.plm_name,
        args.use_4bit,
        args.bnb_4bit_quant_type,
        args.bnb_4bit_compute_dtype,
        args.use_double_quant,
        args.device_map
    )
    dataset = load_dataset(args.dataset_name)
    dataset = dataset.map(lambda example: preprocess_example(example))

    metric = evaluate.load('sacrebleu')

    wandb.login(
        anonymous='never',
        key=WANDB_CLIENT_KEY,
        relogin=True,
        force=True
    )
    wandb.init(project=args.project_name, name=args.run_name)

    lora_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias='none',
        task_type='CAUSAL_LM'
    )

    dataset_size = len(dataset['train'])
    warmup_steps = calculate_warmup_steps(
        epochs=args.num_epochs,
        dataset_size=dataset_size,
        batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        group_by_length=args.group_by_length,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
    )

    def compute_sacrebleu(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"SacreBLEU": result["score"]}
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    # data_collator = DataCollatorForCausalLM(
    #     tokenizer=tokenizer,
    # )

    if args.use_fsdp:
        accelerator = Accelerator()
        trainer = accelerator.prepare(
            SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
                compute_metrics=compute_sacrebleu,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                peft_config=lora_config,
                dataset_text_field='text',
                max_seq_length=args.max_seq_length,
                args=training_args,
                packing=args.packing
            )
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=compute_sacrebleu,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=lora_config,
            dataset_text_field='text',
            max_seq_length=args.max_seq_length,
            args=training_args,
            packing=args.packing
        )
            
    trainer.train()
    
    if args.use_fsdp:
        trainer.model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(trainer.model)
        )
    else:
        trainer.model.save_pretrained(
            args.output_dir
        )


if __name__ == '__main__':
    yaml_path = './llama_config.yaml'
    args = parse_arguments_llama(yaml_path)
    train(args)
