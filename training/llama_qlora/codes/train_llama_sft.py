"""
Train the model with the specified configuration

The following functions are available:
- load_model_and_tokenizer: Load the model and tokenizer with the specified configuration
- calculate_warmup_steps: Calculate the number of warmup steps
- postprocess_text: Postprocess the text
- drop_long_texts: Drop long texts from the dataset
- train: Train the model

Example:
    $ Single GPU: python train_llama_sft.py 
    $ Multi GPU (DDP): accelerate launch --main_process_port 50001 --config_file ../configs/deepspeed_train_config_bf16.yaml train_llama_sft.py

Notes:
- The training arguments are in the llama_config.yaml file
"""
# built-in
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from collections import defaultdict

# third-party
import torch
import wandb
import evaluate
import bert_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from accelerate import Accelerator

# custom
sys.path.append('../../')
from training_utils import set_seed
from argument import load_yaml_config, parse_arguments_llama
from data_collator import CustomDataCollatorForCompletionOnlyLM
sys.path.append('../../../../') # Use your own path
from custom_utils.general_secret import WANDB_CLIENT_KEY # Use your own path


LANG_TABLE = {
    "en": "English",
    "ko": "한국어"
}


def load_model_and_tokenizer(plm_name, device_map, max_length, use_gradient_checkpointing, bnb_config, lora_config):
    """
    Load the model and tokenizer with the specified configuration

    Args:
    - plm_name (str): The name of the pre-trained language model
    - device_map (int): The device map
    - max_length (int): The maximum length of the text
    - use_gradient_checkpointing (bool): Whether to use gradient checkpointing
    - bnb_config (BitsAndBytesConfig): The BitsAndBytesConfig
    - lora_config (LoraConfig): The LoraConfig

    Returns:
    - model (PreTrainedModel): The model
    - tokenizer (PreTrainedTokenizer): The tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained(
        plm_name,
        quantization_config=bnb_config,
        torch_dtype=bnb_config.bnb_4bit_compute_dtype,
        attn_implementation='flash_attention_2',
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.pad_token_id = 2
    model.config.eos_token_id = 46332

    tokenizer = AutoTokenizer.from_pretrained(plm_name, trust_remote_code=True)
    tokenizer.pad_token = "</s>"
    tokenizer.pad_token_id = 2
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 46332
    tokenizer.add_eos_token = True
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = max_length

    src_token, tgt_token = "<|src|>", "<|tgt|>"
    tokenizer.add_tokens([src_token, tgt_token])
    print(f"Added tokens: {src_token}, {tgt_token}")
    print(f"Added token ids: {tokenizer.convert_tokens_to_ids([src_token, tgt_token])}")

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
    """
    Calculate the number of warmup steps

    Args:
    - epochs (int): The number of epochs
    - dataset_size (int): The size of the dataset
    - batch_size (int): The batch size
    - gradient_accumulation_steps (int): The number of gradient accumulation steps
    - warmup_ratio (float): The warmup ratio

    Returns:
    - warmup_steps (int): The number of warmup steps
    """
    steps_per_epoch = (dataset_size / batch_size)
    total_steps = epochs * steps_per_epoch / gradient_accumulation_steps
    total_steps_per_device = total_steps / torch.cuda.device_count()
    warmup_steps = int(total_steps_per_device * warmup_ratio)
    return warmup_steps


def drop_long_texts(dataset, tokenizer, max_length=512):
    """
    Drop long texts from the dataset

    Args:
    - dataset (Dataset): The dataset
    - tokenizer (AutoTokenizer): The tokenizer
    - max_length (int): The maximum length of the text
    - len_threshold (int): The length threshold

    Returns:
    - dataset_dropped (Dataset): The dataset with long texts dropped
    """
    df = pd.DataFrame(dataset)

    rows_to_drop = []
    tqdm_iterator = tqdm(df.iterrows(), total=len(df), desc="Dropping long texts")
    for idx, row in tqdm_iterator:
        src, tgt = row['direction'].split('-')[0], row['direction'].split('-')[1]
        prompt = f"Translate this from {src} to {tgt}."
        suffix_src = f"### {src}:"
        suffix_tgt = f"### {tgt}:"
        text = f"{prompt}\n{suffix_src}\n{row['src']}\n{suffix_tgt} {row['tgt']}"
        outputs = tokenizer.encode_plus(
            text,
            padding=False,
            truncation=True,
            max_length=max_length+1,
            return_tensors='pt',
            return_attention_mask=False,
            return_length=False
        )
        
        input_len = len(outputs.input_ids.squeeze())
        if input_len > max_length:
            rows_to_drop.append(idx)
    
    df_dropped = df.drop(rows_to_drop)
    dataset_dropped = Dataset.from_pandas(df_dropped)

    print(f"Dropped (over {max_length}): {len(rows_to_drop)}")

    return dataset_dropped


def map_bidirectional(dataset):
    mapped_dataset = {}
    for split in dataset.keys():
        split_dataset = defaultdict(list)
        tqdm_iterator = tqdm(
            zip(dataset[split]['ko_ref'], dataset[split]['en_ref']),
            total=len(dataset[split]['ko_ref']), 
            desc=f"Mapping {split} split"
        )
        for ko, en in tqdm_iterator:
            split_dataset['src'].extend([ko, en])
            split_dataset['tgt'].extend([en, ko])
            split_dataset['direction'].extend(['ko-en', 'en-ko'])
            
        mapped_dataset[split] = Dataset.from_dict(split_dataset)
    
    mapped_dataset = DatasetDict(mapped_dataset)

    return mapped_dataset


def add_special_lang_tokens(tokenizer):
    for lang_key in LANG_TABLE.keys():
        lang_token = f'<|{lang_key}|>'
        tokenizer.add_special_tokens({"additional_special_tokens": [lang_token]})
        print(f"Added special lang token: {lang_token}")
        print(f"Added special lang token id: {tokenizer.convert_tokens_to_ids(lang_token)}")


def train(args):
    """
    Train the model

    Args:
    - args (Namespace): The arguments
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
        task_type='CAUSAL_LM'
    )

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model, tokenizer = load_model_and_tokenizer(
        plm_name=args.plm_name, 
        device_map=device_map, 
        max_length=args.max_length, 
        use_gradient_checkpointing=args.gradient_checkpointing,
        bnb_config=bnb_config, 
        lora_config=lora_config
    )

    dataset = load_dataset(args.dataset_name)
    dataset = map_bidirectional(dataset)

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
        logging_steps = min(25, warmup_steps // 10)
        eval_steps = warmup_steps
        save_steps = warmup_steps
        train_dataset = drop_long_texts(dataset['train'], tokenizer)
        eval_dataset = drop_long_texts(dataset['validation'], tokenizer)

    wandb.login(
        anonymous='never',
        key=WANDB_CLIENT_KEY,
        relogin=True,
        force=True
    )
    wandb.init(project=project_name, name=args.run_name)

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

    data_collator=CustomDataCollatorForCompletionOnlyLM(
        instruction_template=f"{args.instruction}\n{args.suffix_src}",
        response_template=f"\n{args.suffix_tgt}",
        tokenizer=tokenizer, 
        lang_table=LANG_TABLE,
        mlm=False
    )

    
    def formatting_func(example):
        output_texts = []
        for src_text, tgt_text, direction in zip(example['src'], example['tgt'], example['direction']):
            src, tgt = direction.split('-')[0], direction.split('-')[1]
            text = f"Translate this from {LANG_TABLE[src]} to {LANG_TABLE[tgt]}.\n### {LANG_TABLE[src]}: {src_text}\n### {LANG_TABLE[tgt]}: {tgt_text}"
            output_texts.append(text)
        return output_texts

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
        
        IGNORE_INDEX = -100

        preds, labels, inputs = p.predictions[0], p.label_ids, p.inputs

        first_non_ignore_indices = np.argmax(labels != IGNORE_INDEX, axis=1)
        for i, idx in enumerate(first_non_ignore_indices):
            preds[i, :idx-1] = tokenizer.pad_token_id
        preds[preds == IGNORE_INDEX] = tokenizer.pad_token_id
        labels = np.where(labels != IGNORE_INDEX, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        signs = []
        for input_id in inputs:
            lang_start_idxs = {}
            for lang in LANG_TABLE.keys():
                lang_encoding = np.array(tokenizer.encode(LANG_TABLE[lang], add_special_tokens=False))
                lang_idx = find_start_idx_window(input_id, lang_encoding)
                lang_start_idxs[lang] = lang_idx
                if len(lang_start_idxs.keys()) == 2:
                    lang1, lang2 = list(lang_start_idxs.keys())
                    sign = f"{lang1}2{lang2}" if lang_start_idxs[lang1] < lang_start_idxs[lang2] else f"{lang2}2{lang1}"
                    signs.append(sign)
                    break

        decodings = defaultdict(list)
        for sign, pred, label in zip(signs, decoded_preds, decoded_labels):
            pred_col, label_col = '_'.join(['pred', sign]), '_'.join(['label', sign])
            decodings[pred_col].append(pred)
            decodings[label_col].append(label)

        random_indices = np.random.choice(len(decoded_labels), 10, replace=False)
        random_decoded_labels = np.array(decoded_labels)[random_indices]
        random_decoded_preds = np.array(decoded_preds)[random_indices]

        for decoded_label, decoded_pred in zip(random_decoded_labels, random_decoded_preds):
            print("[LABEL]")
            print(decoded_label)
            print("[PREDICTION]")
            print(decoded_pred)

        sacrebleu_scores = {
            metrics_result_key('sacrebleu', key): compute_sacrebleu(decodings, key) for key in decodings.keys() if 'pred' in key
        }
        bertscore_scores = {
            metrics_result_key('bertscore', key): compute_bertscore(decodings, key) for key in decodings.keys() if 'pred' in key
        }
        result = {
            **sacrebleu_scores,
            **bertscore_scores
        }
        return result

    trainer = SFTTrainer(
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
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    yaml_path = '../configs/llama_config.yaml'
    args = parse_arguments_llama(yaml_path)
    acc_yaml_path = '../configs/deepspeed_train_config_bf16.yaml'
    acc_config = load_yaml_config(acc_yaml_path)

    args.per_device_batch_size = acc_config['deepspeed_config']['train_micro_batch_size_per_gpu']
    args.gradient_accumulation_steps = acc_config['deepspeed_config']['gradient_accumulation_steps']
    args.max_grad_norm = acc_config['deepspeed_config']['gradient_clipping']
    args.dataloader_num_workers = acc_config['num_processes']

    train(args)
