# built-in
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
from collections import defaultdict

# third-party
import torch
import numpy as np
import wandb
import evaluate
import bert_score
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator

# custom
sys.path.append('../../')
from training_utils import set_seed
from argument import parse_arguments_mt5
sys.path.append('../../../../') # Use your own path
from custom_utils.general_secret import WANDB_CLIENT_KEY # Use your own path


def load_model_and_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def set_src2tgt(example, idx, mode):
    if mode.startswith('mixed'):
        example.update({'src2tgt': idx % 2 == 0})
    elif mode.startswith('en2ko'):
        example.update({'src2tgt': True})
    elif mode.startswith('ko2en'):
        example.update({'src2tgt': False})
    return example


def add_special_tokens_by_mode(tokenizer, src_sign=None, tgt_sign=None):
    if src_sign is not None:
        tokenizer.add_special_tokens({'additional_special_tokens': [src_sign]})
        print("Source sign:", src_sign)
        print("Source sign id:", tokenizer(src_sign)['input_ids'][0])
    if tgt_sign is not None:
        tokenizer.add_special_tokens({'additional_special_tokens': [tgt_sign]})
        print("Target sign:", tgt_sign)
        print("Target sign id:", tokenizer(tgt_sign)['input_ids'][0])


def preprocess_example(example, mode, max_length, tokenizer):
    if example['src2tgt']:
        src_col, tgt_col = 'en', 'ko'
    else:
        src_col, tgt_col = 'ko', 'en'
    
    if 'clean' in mode:
        src_sign = None
        tgt_sign = None
        src_text = example[src_col]
        tgt_text = example[tgt_col]
    elif 'separate' in mode:
        src_sign = f'<{src_col}>'
        tgt_sign = f'<{tgt_col}>'
        src_text = ' '.join([src_sign, example[src_col]])
        tgt_text = ' '.join([tgt_sign, example[tgt_col]])
    elif 'first' in mode:
        src_sign = f'<{src_col}2{tgt_col}>'
        tgt_sign = None
        src_text = ' '.join([src_sign, example[src_col]])
        tgt_text = example[tgt_col]
    elif 'second' in mode:
        src_sign = None
        tgt_sign = f'<{src_col}2{tgt_col}>'
        src_text = example[src_col]
        tgt_text = ' '.join([tgt_sign, example[tgt_col]])

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


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
    steps_per_epoch = (dataset_size / batch_size)
    total_steps = epochs * steps_per_epoch / gradient_accumulation_steps
    total_steps_per_device = total_steps / torch.cuda.device_count()
    warmup_steps = int(total_steps_per_device * warmup_ratio)
    return warmup_steps


def train(args):
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.plm_name)
    if args.translation_mode.endswith('token'):
        if 'separate' in args.translation_mode:
            src_sign, tgt_sign = '<en>', '<ko>'
        elif 'first' in args.translation_mode:
            if args.translation_mode.startswith('en2ko'):
                src_sign, tgt_sign = '<en2ko>', None
            elif args.translation_mode.startswith('ko2en'):
                src_sign, tgt_sign = '<ko2en>', None
            elif args.translation_mode.startswith('mixed'):
                src_sign, tgt_sign = '<en2ko>', '<ko2en>'
        elif 'second' in args.translation_mode:
            if args.translation_mode.startswith('en2ko'):
                src_sign, tgt_sign = None, '<en2ko>'
            elif args.translation_mode.startswith('ko2en'):
                src_sign, tgt_sign = None, '<ko2en>'
            elif args.translation_mode.startswith('mixed'):
                src_sign, tgt_sign = '<en2ko>', '<ko2en>'

        add_special_tokens_by_mode(tokenizer, src_sign, tgt_sign)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    dataset = load_dataset(args.dataset_name)
    dataset = dataset.map(lambda example, idx: set_src2tgt(example, idx, args.translation_mode), with_indices=True)
    dataset = dataset.map(lambda example: preprocess_example(example, args.translation_mode, args.max_length, tokenizer))

    metric = evaluate.load('sacrebleu')

    warmup_steps = calculate_warmup_steps(
        epochs=args.num_epochs,
        dataset_size=len(dataset['train']),
        batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio
    )
    print("Warmup steps = Eval steps = Save steps =", warmup_steps)

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
        eval_dataset = dataset['validation'].select(range(1000))

    wandb.login(
        anonymous='never',
        key=WANDB_CLIENT_KEY,
        relogin=True,
        force=True
    )
    wandb.init(project=project_name, name=args.run_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_epochs,
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
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        include_inputs_for_metrics=args.include_inputs_for_metrics,
    )

    def clean_key(key):
        return key.split('_')[1]

    def metrics_result_key(prefix, key):
        return f"{prefix}_{clean_key(key)}"
    
    def compute_sacrebleu(decodings, key):
        return metric.compute(predictions=decodings[f'pred_{clean_key(key)}'], references=decodings[f'label_{clean_key(key)}'])['score']
    
    def compute_bertscore(decodings, key):
        lang = clean_key(key).split('2')[1]
        return bert_score.score(cands=decodings[f'pred_{clean_key(key)}'], refs=decodings[f'label_{clean_key(key)}'], lang=lang)[2].mean().item() * 100

    def compute_metrics(p):
        preds, labels, inputs = p.predictions[0], p.label_ids, p.inputs

        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        if args.translation_mode.startswith('mixed'):
            if args.translation_mode.endswith('token'):
                if 'separate' in args.translation_mode:
                    signs = []
                    for input, label in zip(inputs, labels):
                        input_sign = tokenizer.convert_ids_to_tokens([input[0]])[0][1:-1]
                        label_sign = tokenizer.convert_ids_to_tokens([label[0]])[0][1:-1]
                        sign = f"{input_sign}2{label_sign}"
                        signs.append(sign)
                elif 'first' in args.translation_mode:
                    signs = [tokenizer.convert_ids_to_tokens([input[0]])[0][1:-1] for input in inputs]
                elif 'second' in args.translation_mode:
                    signs = [tokenizer.convert_ids_to_tokens([label[0]])[0][1:-1] for label in labels]
            else:
                signs = [decoded_input.split(' ')[0][1:-1] for decoded_input in decoded_inputs]
        else:
            signs = [args.translation_mode.split('-')[0]] * len(decoded_labels)

        decodings = defaultdict(list)
        for sign, pred, label in zip(signs, decoded_preds, decoded_labels):
            pred_col, label_col = '_'.join(['pred', sign]), '_'.join(['label', sign])
            decodings[pred_col].append(pred)
            decodings[label_col].append(label)

        random_indices = np.random.choice(len(decoded_labels), 10, replace=False)
        for decoded_label, decoded_pred in zip(np.array(decoded_labels)[random_indices], np.array(decoded_preds)[random_indices]):
            print("[LABEL]")
            print(decoded_label)
            print("[PREDICTION]")
            print(decoded_pred)

        sacrebleu_scores = {
            metrics_result_key('SacreBLEU', key): compute_sacrebleu(decodings, key) for key in decodings.keys() if 'pred' in key
        }
        
        bertscore_scores = {
            metrics_result_key('BERTScore', key): compute_bertscore(decodings, key) for key in decodings.keys() if 'pred' in key
        }
        
        result = {**sacrebleu_scores, **bertscore_scores}

        return result

    if args.use_fsdp:
        # accelerate config: /home/tmax/.cache/huggingface/accelerate/default_config.yaml
        accelerator = Accelerator()
        trainer = accelerator.prepare(
            Seq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=default_data_collator,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    yaml_path = './mt5_config.yaml'
    args = parse_arguments_mt5(yaml_path)
    train(args)