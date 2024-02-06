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
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
import torch.distributed.checkpoint as DCP

# custom
sys.path.append('../../../../')
from custom_utils.training_utils import set_seed
from custom_utils.general_secret import WANDB_CLIENT_KEY
from custom_utils.argument import parse_arguments_mbart


def load_model_and_tokenizer(model_name, src_lang='en_XX', tgt_lang='ko_KR'):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
    return model, tokenizer


def preprocess_example(example, max_length, tokenizer):
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
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
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

    model, tokenizer = load_model_and_tokenizer(args.plm_name, args.src_lang, args.tgt_lang)
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    dataset = load_dataset(args.dataset_name)
    dataset = dataset.map(lambda example: preprocess_example(example, args.max_length, tokenizer))

    metric = evaluate.load('sacrebleu')

    wandb.login(
        anonymous='never',
        key=WANDB_CLIENT_KEY,
        relogin=True,
        force=True
    )
    wandb.init(project=args.project_name, name=args.run_name)

    warmup_steps = calculate_warmup_steps(
        epochs=args.num_train_epochs,
        dataset_size=len(dataset['train']),
        batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps, # 전체 step의 10%
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        eval_steps=warmup_steps, # 전체 step의 10%
        save_steps=warmup_steps, # 전체 step의 10%
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
    )

    def compute_sacrebleu(p):
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

    if args.use_fsdp:
        accelerator = Accelerator()
        trainer = accelerator.prepare(
            Seq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
                data_collator=default_data_collator,
                compute_metrics=compute_sacrebleu,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=default_data_collator,
            compute_metrics=compute_sacrebleu,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

    trainer.train()
    trainer.save_model(args.output_dir)

    state_dict_path = os.path.join('../models', args.run_name)
    merge_and_save_distcp(trainer.model, state_dict_path)


if __name__ == '__main__':
    yaml_path = './mbart_config.yaml'
    args = parse_arguments_mbart(yaml_path)
    train(args)
