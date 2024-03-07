# built-in
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
from collections import defaultdict

# third-party
import torch
import numpy as np
import wandb
import evaluate
import bert_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from peft import get_peft_model

# custom
sys.path.append('./')
sys.path.append('../../')
from training_utils import set_seed
from argument import parse_arguments_mt5_qlora
sys.path.append('../../../../') # Use your own path
from custom_utils.general_secret import WANDB_CLIENT_KEY # Use your own path


def load_model_and_tokenizer(plm_name, device_map, bnb_config, lora_config):
    max_memory = '24000MB'
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    device_map = args.device_map

    if os.environ.get('LOCAL_RANK') is not None:
        print("LOCAL_RANK:", os.environ.get('LOCAL_RANK'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    model = AutoModelForSeq2SeqLM.from_pretrained(
        plm_name,
        quantization_config=bnb_config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=bnb_config.bnb_4bit_compute_dtype,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(plm_name)

    return model, tokenizer


def set_src2tgt(example, idx, mode):
    if mode.startswith('mixed'):
        example.update({'src2tgt': idx % 2 == 0})
    elif mode.startswith('en2ko'):
        example.update({'src2tgt': True})
    elif mode.startswith('ko2en'):
        example.update({'src2tgt': False})
    return example


def src_and_tgt_by_mode(mode):
    src_sign, tgt_sign = None, None
    if 'separate' in mode:
        src_sign, tgt_sign = '<en>', '<ko>'
    elif 'first' in mode:
        src_sign, tgt_sign = '<2ko>', None
    elif 'second' in mode:
        src_sign, tgt_sign = None, '<2ko>'
    return src_sign, tgt_sign


def change_sign(sign):
    if sign is None:
        return sign
    if 'en' in sign:
        sign = sign.replace('en', 'ko')
    elif 'ko' in sign:
        sign = sign.replace('ko', 'en')
    return sign


def add_special_token(tokenizer, sign):
    tokenizer.add_special_tokens({'additional_special_tokens': [sign]})
    print("Special token:", sign)
    print("Special token id:", tokenizer(sign)['input_ids'][0])


def add_special_tokens_by_mode(tokenizer, mode, src_sign=None, tgt_sign=None):
    src_sign, tgt_sign = src_and_tgt_by_mode(mode)
    if src_sign is not None:
        add_special_token(tokenizer, src_sign)
        if mode.startswith('mixed'):
            add_special_token(tokenizer, change_sign(src_sign))
    if tgt_sign is not None:
        add_special_token(tokenizer, tgt_sign)
        if mode.startswith('mixed'):
            add_special_token(tokenizer, change_sign(tgt_sign))


def preprocess_example(example, mode, max_length, tokenizer):
    src_col, tgt_col = 'en', 'ko'
    src_sign, tgt_sign = src_and_tgt_by_mode(mode)
    if not example['src2tgt']:
        src_col, tgt_col = tgt_col, src_col
        src_sign, tgt_sign = change_sign(src_sign), change_sign(tgt_sign)
    
    if 'clean' in mode:
        src_text = example[src_col]
    elif 'separate' in mode:
        src_text = f"{src_sign} {example[src_col]} {tgt_sign}"
    elif 'first' in mode:
        src_text = f"{src_sign} {example[src_col]}"
    elif 'second' in mode:
        src_text = f"{example[src_col]} {tgt_sign}"

    tgt_text = example[tgt_col]

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


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.use_4bit else (bnb.nn.Linear8bitLt if args.use_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def train(args):
    set_seed(args.seed)

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_double_quant,
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
        device_map=args.device_map,
        bnb_config=bnb_config,
        lora_config=lora_config,
    )

    add_special_tokens_by_mode(tokenizer, args.translation_mode)

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
    yaml_path = os.path.join(SCRIPT_DIR, './mt5_qlora_config.yaml')
    args = parse_arguments_mt5_qlora(yaml_path)
    train(args)
