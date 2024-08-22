# built-in
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import re
import sys
import warnings
from collections import defaultdict

# third-party
import torch
import wandb
import evaluate
import bert_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from accelerate import Accelerator
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
# from transformers import logging # 'gamma', 'beta' 관련 로그 출력 방지 (progress bar 출력 안됨 주의)
# logging.set_verbosity_error()

# custom
sys.path.append(os.path.join(SCRIPT_DIR, '../../'))
from training_utils import set_seed
from argument import load_yaml_config, parse_arguments_llama
from trainer import SFTTrainerWithEosToken
from data_collator import Llama2DataCollatorForCompletionOnlyLM, Llama3DataCollatorForCompletionOnlyLM
sys.path.append(os.path.join(SCRIPT_DIR, '../../../../'))
try:
    from custom_utils.general_secret import WANDB_CLIENT_KEY # Use your own path
except:
    WANDB_CLIENT_KEY = "YOUR_WANDB_CLIENT_KEY"


LANG_TABLE = {
    "ko": "한국어",
    "en": "English",
    "ja": "日本語",
    "zh": "中文",
}
HF_MODEL_CACHE_DIR = "/data/.cache/hub/"
HF_DATASETS_CACHE_DIR = "/data/.cache/datasets/"


def load_model_and_tokenizer(args):
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=args.use_double_quant
        )
    else:
        quantization_config = None

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    compute_dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.plm_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        attn_implementation='flash_attention_2',
        device_map=device_map,
        cache_dir=HF_MODEL_CACHE_DIR,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if args.use_lora:
        modules = args.lora_target_modules
        print("LoRA adapted modules:", modules)
        print("LoRA target layers:", args.lora_target_layers.upper())
        if args.lora_target_layers == 'all':
            target_layer_indices = [i for i in range(len(model.model.layers))]
        elif args.lora_target_layers == 'odd':
            target_layer_indices = [i for i in range(len(model.model.layers)) if i % 2 == 1]
        elif args.lora_target_layers == 'even':
            target_layer_indices = [i for i in range(len(model.model.layers)) if i % 2 == 0]
        
        if args.use_mora:
            lora_config = LoraConfig(
                use_mora=args.use_mora,
                mora_type=args.mora_type,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias='none',
                target_modules=modules,
                task_type='CAUSAL_LM',
                layers_to_transform=target_layer_indices,
            )
        else:
            lora_config = LoraConfig(
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias='none',
                target_modules=modules,
                task_type='CAUSAL_LM',
                layers_to_transform=target_layer_indices,
            )
        
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name, trust_remote_code=True)
    tokenizer.eos_token_id = args.eos_token_id
    tokenizer.pad_token_id = args.pad_token_id
    print(f"PAD token: {tokenizer.pad_token}")
    print(f"PAD token id: {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token id: {tokenizer.eos_token_id}")
    tokenizer.add_eos_token = True
    tokenizer.padding_side = 'left' # Decoder 기반 모델의 경우 left가 맞음
    tokenizer.model_max_length = args.max_length

    src_token, tgt_token = "<|src|>", "<|tgt|>"
    tokenizer.add_tokens([src_token, tgt_token])
    print(f"Added tokens: {src_token}, {tgt_token}")
    print(f"Added token ids: {tokenizer.convert_tokens_to_ids([src_token, tgt_token])}")

    return model, tokenizer


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
    steps_per_epoch = (dataset_size / batch_size)
    total_steps = epochs * steps_per_epoch / gradient_accumulation_steps
    total_steps_per_device = total_steps / torch.cuda.device_count()
    warmup_steps = int(total_steps_per_device * warmup_ratio)
    warmup_steps = warmup_steps if warmup_steps > 1 else 0
    return warmup_steps


def map_bidirectional(dataset, lang_col1='ko', lang_col2='en', exist_direction=False):
    mapped_dataset = defaultdict(list)
    if exist_direction:
        zip_for_iter = zip(dataset[lang_col1], dataset[lang_col2], dataset['direction'])
    else:
        zip_for_iter = zip(dataset[lang_col1], dataset[lang_col2])

    tqdm_iterator = tqdm(
        zip_for_iter,
        total=len(dataset), 
        desc=f"Mapping dataset"
    )
    for lang1, lang2, dir in tqdm_iterator:
        mapped_dataset['src'].extend([lang1, lang2])
        mapped_dataset['tgt'].extend([lang2, lang1])
        lang_col1 = lang_col1.replace('_ref', '') if lang_col1.endswith('_ref') else lang_col1
        lang_col2 = lang_col2.replace('_ref', '') if lang_col2.endswith('_ref') else lang_col2
        if exist_direction:
            src, tgt = dir.split('-')
            mapped_dataset['direction'].extend([f'{src}-{tgt}', f'{tgt}-{src}'])
        else:
            mapped_dataset['direction'].extend([f'{lang_col1}-{lang_col2}', f'{lang_col2}-{lang_col1}'])
        
    mapped_dataset = Dataset.from_dict(mapped_dataset)

    return mapped_dataset


def mix_word_dataset(dataset, word_dataset, word_size=10000):
    sent_df = pd.DataFrame(dataset)
    word_df = pd.DataFrame(word_dataset)
    try:
        word_df = word_df.sample(word_size).reset_index(drop=True)
    except:
        warnings.warn("Word dataset size is smaller than the specified size. Using the original size.")

    total_df = pd.concat([sent_df, word_df], axis=0)
    total_df = total_df.sample(frac=1).reset_index(drop=True)

    dataset = Dataset.from_pandas(total_df)
    
    return dataset


def add_special_lang_tokens(tokenizer):
    for lang_key in LANG_TABLE.keys():
        lang_token = f'<|{lang_key}|>'
        tokenizer.add_special_tokens({"additional_special_tokens": [lang_token]})
        print(f"Added special lang token: {lang_token}")
        print(f"Added special lang token id: {tokenizer.convert_tokens_to_ids(lang_token)}")


def train(args):
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)

    if args.mix_word_dataset:
        train_dataset = load_dataset(args.train_dataset_name, cache_dir=HF_DATASETS_CACHE_DIR)['train']
        eval_dataset = load_dataset(args.eval_dataset_name, cache_dir=HF_DATASETS_CACHE_DIR)['validation']
        train_word_dataset = load_dataset(args.train_word_dataset_name, cache_dir=HF_DATASETS_CACHE_DIR)['train']
        # eval_word_dataset = load_dataset(args.eval_word_dataset_name, cache_dir=HF_DATASETS_CACHE_DIR)['validation']
        train_dataset = mix_word_dataset(train_dataset, train_word_dataset, word_size=args.train_word_size)
        # eval_dataset = mix_word_dataset(eval_dataset, eval_word_dataset, word_size=len(eval_word_dataset))
    else:
        train_dataset = load_dataset(args.train_dataset_name, cache_dir=HF_DATASETS_CACHE_DIR)['train']
        eval_dataset = load_dataset(args.eval_dataset_name, cache_dir=HF_DATASETS_CACHE_DIR)['validation']

    train_dataset = map_bidirectional(train_dataset, lang_col1=args.lang_col1, lang_col2=args.lang_col2, exist_direction=True)
    eval_dataset = map_bidirectional(eval_dataset, lang_col1=args.lang_col1, lang_col2=args.lang_col2, exist_direction=True)

    if args.just_test:
        train_dataset = train_dataset.select(range(1000))
        eval_dataset = eval_dataset.select(range(10))

    dataset_size = len(train_dataset)
    warmup_steps = calculate_warmup_steps(
        epochs=args.num_epochs,
        dataset_size=dataset_size,
        batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio
    )
    print(f"Warmup steps: {warmup_steps}")

    if args.just_test:
        project_name = 'test'
        output_dir = '/data/sehyeong/nmt/models/test'
        logging_steps = 1
        eval_steps = 1
        save_steps = 1
    else:
        project_name = args.project_name
        output_dir = args.output_dir
        logging_steps = min(25, warmup_steps // 10) if warmup_steps >= 10 else args.logging_steps
        eval_steps = warmup_steps if warmup_steps > 0 else args.eval_steps
        # eval_steps = args.eval_steps
        save_steps = warmup_steps if warmup_steps > 0 else args.save_steps
        # save_steps = args.save_steps

    args.project_name = project_name
    args.output_dir = output_dir
    args.warmup_steps = warmup_steps
    args.logging_steps = logging_steps
    args.eval_steps = eval_steps
    args.save_steps = save_steps

    wandb.login(
        anonymous='never',
        key=WANDB_CLIENT_KEY,
        relogin=True,
        force=True
    )
    wandb.init(project=project_name, name=args.run_name, config=args)

    training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size // 2, # bertscore 계산 시 메모리 부족 방지
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
        include_inputs_for_metrics=True,
    )


    def formatting_func(example):
        prompts = []
        for src_text, tgt_text, direction in zip(example['src'], example['tgt'], example['direction']):
            src_langcode, tgt_langcode = direction.split('-')[0], direction.split('-')[1]
            src_lang, tgt_lang = LANG_TABLE[src_langcode], LANG_TABLE[tgt_langcode]
            
            instruction_part = {
                'head': "<instruction>",
                'body': f"Translate the source sentence from {src_lang} to {tgt_lang}.\nBe sure to reflect the guidelines below when translating.",
                'tail': "</instruction>"
            }
            instruction = f"{instruction_part['head']}\n{instruction_part['body']}\n{instruction_part['tail']}"
            guideline_part = {
                'head': "<guideline>",
                'body': [
                    "Translate plainly."
                ],
                'tail': "</guideline>"
            }
            guideline_body_part = '\n'.join([f'- {body}' for body in guideline_part['body']])
            guideline = f"{guideline_part['head']}\n{guideline_body_part}\n{guideline_part['tail']}"
            src_part = {
                'head': f"<source><{src_lang}>",
                'body': src_text.strip(),
                'tail': f"</{src_lang}></source>"
            }
            src = f"{src_part['head']}\n{src_part['body']}\n{src_part['tail']}"
            tgt_part = {
                'head': f"<target><{tgt_lang}>",
                'body': tgt_text.strip(),
                'tail': f"</{tgt_lang}></target>"
            }
            tgt = f"{tgt_part['head']}\n{tgt_part['body']}\n{tgt_part['tail']}"
            translation_part = {
                'head': "<translation>",
                'body': f"{src}\n{tgt}",
                'tail': "</translation>"
            }
            translation = f"{translation_part['head']}\n{translation_part['body']}\n{translation_part['tail']}"
            
            prompt = f"{instruction}\n\n{guideline}\n\n{translation}"

            prompts.append(prompt)

        return prompts

    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def compute_metrics(p):
        """
        Compute the SacreBLEU score

        Args:
        - p (EvalPrediction): The evaluation prediction

        Returns:
        - result (dict): The result
        """
        def find_start_idx_window(long_array, short_array):
            for i in range(len(long_array) - len(short_array) + 1):
                if np.array_equal(long_array[i:i+len(short_array)], short_array):
                    return i
            return -1

        def clean_key(key):
            return key.split('_')[1]

        def metrics_result_key(prefix, key):
            return f"{prefix}_{clean_key(key)}"

        def compute_sacrebleu(decodings, key):
            metric = evaluate.load('sacrebleu')
            return metric.compute(
                predictions=decodings[f'pred_{clean_key(key)}'], 
                references=decodings[f'label_{clean_key(key)}']
            )['score']
        
        def compute_bertscore(decodings, key):
            lang = clean_key(key).split('2')[1]
            return bert_score.score(
                cands=decodings[f'pred_{clean_key(key)}'], 
                refs=decodings[f'label_{clean_key(key)}'], 
                lang=lang,
                batch_size=4
            )[2].mean().item() * 100
        
        def get_lang_signs(inputs):
            signs = []
            for input_id in inputs:
                src_lang, tgt_lang = None, None
                for lang_code, lang_full in LANG_TABLE.items():
                    lang_suffix_src = f'<source><{lang_full}>\n'
                    lang_encoding_src = np.array(tokenizer.encode(lang_suffix_src, add_special_tokens=False))
                    lang_idx_src = find_start_idx_window(input_id, lang_encoding_src)
                    if lang_idx_src != -1:
                        src_lang = lang_code
                        continue
                    lang_suffix_tgt = f'<target><{lang_full}>\n'
                    lang_encoding_tgt = np.array(tokenizer.encode(lang_suffix_tgt, add_special_tokens=False))
                    lang_idx_tgt = find_start_idx_window(input_id, lang_encoding_tgt)
                    if lang_idx_tgt != -1:
                        tgt_lang = lang_code
                        continue
                    if src_lang is not None and tgt_lang is not None:
                        break
                sign = f"{src_lang}2{tgt_lang}"
                signs.append(sign)
            return signs
        
        IGNORE_INDEX = -100
        preds, labels, inputs = p.predictions[0], p.label_ids, p.inputs

        first_non_ignore_indices = np.argmax(labels != IGNORE_INDEX, axis=1)
        for i, non_ig_idx in enumerate(first_non_ignore_indices):
            if non_ig_idx > 0:
                preds[i, :non_ig_idx-1] = IGNORE_INDEX
        eos_indices = np.argmax(preds == tokenizer.eos_token_id, axis=1)
        for i, eos_idx in enumerate(eos_indices):
            if eos_idx < len(preds[i]):
                preds[i, eos_idx+1:] = IGNORE_INDEX

        preds = np.where(preds == IGNORE_INDEX, tokenizer.pad_token_id, preds)
        labels = np.where(labels == IGNORE_INDEX, tokenizer.pad_token_id, labels)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [label.strip() for label in decoded_labels]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]

        signs = get_lang_signs(inputs)

        decodings = defaultdict(list)
        for sign, pred, label in zip(signs, decoded_preds, decoded_labels):
            pred_col, label_col = '_'.join(['pred', sign]), '_'.join(['label', sign])
            decodings[pred_col].append(pred)
            decodings[label_col].append(label)

        random_indices = np.random.choice(len(decoded_labels), 10, replace=False)
        random_decoded_labels = np.array(decoded_labels)[random_indices]
        random_decoded_preds = np.array(decoded_preds)[random_indices]

        for decoded_label, decoded_pred in zip(random_decoded_labels, random_decoded_preds):
            print(f"[LABEL]\n{decoded_label}")
            print(f"[PREDICTION]\n{decoded_pred}\n")

        decodings_for_metrics = {key: [re.sub(r'</\w+>', '', text).strip() for text in texts] for key, texts in decodings.items()}
        sacrebleu_scores = {
            metrics_result_key('sacrebleu', key): compute_sacrebleu(decodings_for_metrics, key) for key in decodings_for_metrics.keys() if 'pred' in key
        }
        bertscore_scores = {
            metrics_result_key('bertscore', key): compute_bertscore(decodings_for_metrics, key) for key in decodings_for_metrics.keys() if 'pred' in key
        }
        result = {
            **sacrebleu_scores,
            **bertscore_scores
        }
        return result


    if any([plm_name in args.plm_name.lower() for plm_name in ['llama-3', 'llama3']]):
        data_collator_cls = Llama3DataCollatorForCompletionOnlyLM
        trainer_cls = SFTTrainerWithEosToken
    elif any([plm_name in args.plm_name.lower() for plm_name in ['gemma']]):
        data_collator_cls = Llama3DataCollatorForCompletionOnlyLM
        trainer_cls = SFTTrainer
    elif any([plm_name in args.plm_name.lower() for plm_name in ['llama-2', 'llama2']]):
        data_collator_cls = Llama2DataCollatorForCompletionOnlyLM
        trainer_cls = SFTTrainer
    else:
        raise ValueError("Unknown PLM name")

    data_collator = data_collator_cls(
        instruction_template=args.suffix_instruction,
        response_template=args.suffix_response,
        tokenizer=tokenizer, 
        lang_table=LANG_TABLE,
        mlm=False
    )

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        formatting_func=formatting_func,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        max_seq_length=args.max_length,
        packing=False
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    yaml_path = '../configs/llama_config.yaml'
    args = parse_arguments_llama(os.path.join(SCRIPT_DIR, yaml_path))
    acc_yaml_path = '../configs/deepspeed_train_config_bf16.yaml'
    acc_config = load_yaml_config(os.path.join(SCRIPT_DIR, acc_yaml_path))

    args.per_device_batch_size = acc_config['deepspeed_config']['train_micro_batch_size_per_gpu']
    args.gradient_accumulation_steps = acc_config['deepspeed_config']['gradient_accumulation_steps']
    args.max_grad_norm = acc_config['deepspeed_config']['gradient_clipping']

    train(args)
