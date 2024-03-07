import os
import sys
os.environ["HF_DATASETS_CACHE"] = "/home/tmax/.cache/huggingface/datasets/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "../../"))

import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

from argument import parse_arguments_llama_dpo


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        if os.path.isfile(pytorch_model_path) and torch.distributed.get_rank() == 0:
            os.remove(pytorch_model_path)
            # create an empty toy file to avoid error in deleting old checkpoints
            open(pytorch_model_path, 'w').close()
        return control


LANG_TABLE = {
    "en": "English",
    "de": "German",
    "cs": "Czech",
    "is": "Icelandic",
    "zh": "Chinese",
    "ru": "Russian",
}


def get_necessary_datasets(pairs, data_path, cache_dir=None, use_auth_token=None, streaming=False):
    pairs = set(pairs)
    seen = set()
    raw_data = {}
    for pair in pairs:
        src_lang, tgt_lang = pair.split('-')
        first_lang = src_lang if src_lang != 'en' else tgt_lang
        second_lang = 'en'
        pair_fixed = f"{first_lang}-{second_lang}"
        if (first_lang, second_lang) not in seen:
            raw_data[pair_fixed] = load_dataset(
                data_path,
                pair_fixed,
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
                streaming=streaming,
            )
        seen.add((first_lang, second_lang))
    return raw_data


def get_prompt(src_lang, tgt_lang, ex):
    src_fullname = LANG_TABLE[src_lang]
    tgt_fullname = LANG_TABLE[tgt_lang]

    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
    suffix = f"\n{tgt_fullname}: "

    prompt = prefix + ex[src_lang] + suffix
    return prompt


def get_chosen_rejected(example, tgt_lang, cpo_scorer):
    sys1, sys2, ref = 'gpt4', 'alma', 'ref'

    sys1_score_key = f"{sys1}_{tgt_lang}_{cpo_scorer}"
    sys2_score_key = f"{sys2}_{tgt_lang}_{cpo_scorer}"
    ref_score_key = f"{ref}_{tgt_lang}_{cpo_scorer}"

    sys1_output_key = f"{sys1}_{tgt_lang}"
    sys2_output_key = f"{sys2}_{tgt_lang}"
    ref_output_key = tgt_lang

    if 'Delta' in example and example['Delta'] != 0:
        if example['Delta'] > 0:
            return example[sys1_output_key], example[sys2_output_key]
        else:
            return example[sys2_output_key], example[sys1_output_key]
    
    sentences = [example[ref_output_key], example[sys1_output_key], example[sys2_output_key]]
    scores = [example[ref_score_key], example[sys1_score_key], example[sys2_score_key]]

    score_argmax = scores.index(max(scores))
    score_argmin = scores.index(min(scores))

    best_sentence = sentences[score_argmax]
    worst_sentence = sentences[score_argmin]

    return best_sentence, worst_sentence


def meet_requirements(prompt_tokens, example, tgt_lang, max_src_length):
    if len(prompt_tokens) > max_src_length:
        return False
    if 'required_directions' in example and example['required_directions'] != '':
        tgt = example['required_directions'].split('-')[-1]
        if tgt != tgt_lang:
            return False
    return True


def preprocess_cpo_data(
        data_path,
        pairs,
        tokenizer,
        args,
        training_args,
        split='train'
    ):
    max_src_length = args.max_src_length

    def get_cpo_prompt(examples):
        new_examples = {
            'prompt': [],
            'chosen': [],
            'rejected': [],
        }
        for ex in examples['translation']:
            src_lang, tgt_lang = ex['language_pair'].split('-')
            pair = f"{src_lang}-{tgt_lang}"
            pair_reverse = f"{tgt_lang}-{src_lang}"
            if pair in pairs:
                prompt = get_prompt(src_lang, tgt_lang, ex)
                prompt_tokens = tokenizer(prompt, max_length=max_src_length, padding=True, truncation=True, add_special_tokens=False)['input_ids']
                if not meet_requirements(prompt_tokens, ex, tgt_lang, max_src_length):
                    continue
                chosen, rejected = get_chosen_rejected(ex, tgt_lang, args.cpo_scorer)
                new_examples['prompt'].append(prompt)
                new_examples['chosen'].append(chosen)
                new_examples['rejected'].append(rejected)
        
            if pair_reverse in pairs:
                prompt = get_prompt(tgt_lang, src_lang, ex)
                prompt_tokens = tokenizer(prompt, max_length=max_src_length, padding=True, truncation=True, add_special_tokens=False)['input_ids']
                if not meet_requirements(prompt_tokens, ex, src_lang, max_src_length):
                    continue
                chosen, rejected = get_chosen_rejected(ex, src_lang, args.cpo_scorer)
                new_examples['prompt'].append(prompt)
                new_examples['chosen'].append(chosen)
                new_examples['rejected'].append(rejected)

        return new_examples
    
    raw_data = get_necessary_datasets(pairs, data_path)
    processed_datasets = []
    for pair, sub_raw_data in raw_data.items():
        dataset = sub_raw_data[split]
        if args.max_train_samples is not None:
            max_train_samples = min(len(dataset), max_train_samples)
            dataset = dataset.select(range(args.max_train_samples))
        with training_args.main_process_first(desc="CPO train dataset mapping & preprocessing"):
            if not args.streaming:
                dataset = dataset.map(
                    get_cpo_prompt,
                    batched=True,
                    batch_size=args.per_device_batch_size,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=['translation'],
                    cache_file_name=f"{os.environ['HF_DATASETS_CACHE']}/{args.plm_name.split('/')[-1]}-train-mmt-{pair}-{pairs}-cpo",
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running CPO preprocessing",
                )
            else:
                dataset = dataset.map(
                    get_cpo_prompt,
                    batched=True,
                    batch_size=args.per_device_train_batch_size,
                    num_proc=args.preprocessing_num_workers,
                    desc="Running CPO preprocessing",
                )
        processed_datasets.append(dataset)
    
    datasets = concatenate_datasets(processed_datasets)
    datasets = datasets.shuffle(seed=args.seed)

    return datasets


def main():
    yaml_path = './llama_dpo_config.yaml'
    args = parse_arguments_llama_dpo(yaml_path)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio, # 전체 step의 10%
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
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps, # 전체 step의 10%
        save_steps=args.save_steps, # 전체 step의 10%
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
    train_datasets = preprocess_cpo_data(args.dataset_name, args.language_pairs, tokenizer, args, training_args)
    train_sample = pd.DataFrame(train_datasets).sample(n=10)
    print(train_sample.head())
    # train_sample.to_csv('./cpo_sample.csv', index=False)


if __name__ == "__main__":
    main()
