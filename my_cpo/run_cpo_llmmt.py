import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import sys
import json

import datasets
import torch
from datasets import load_dataset

import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from ojt_translation.my_cpo.utils.cpo_utils import preprocess_cpo_data, load_tokenizer, load_model, SavePeftModelCallback
from ojt_translation.my_cpo.utils.cpo_arguments import ModelArguments, DataTrainingArguments
from utils.cpo_trainer import CPOTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    pairs = set(data_args.language_pairs.split(","))
    train_raw_data, valid_raw_data, test_raw_data = None, None, None
    seen = set()
    
    train_raw_data = {}
    for pair in pairs:
        src_lang, tgt_lang = pair.split("-")
        first_lang = src_lang if src_lang != "en" else tgt_lang
        second_lang = "en"
        if (first_lang, second_lang) not in seen and training_args.do_train:
            train_raw_data[f"{first_lang}-{second_lang}"] = load_dataset(
                data_args.cpo_data_path,
                f"{first_lang}-{second_lang}",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
                )
        seen.add((first_lang, second_lang))
    
    set_seed(training_args.seed)
    tokenizer = load_tokenizer(data_args, model_args, training_args, logger)

    shots_eval_dict = {}
    if data_args.few_shot_eval_path:
        for lg_pair in test_raw_data.keys():
            pair_shot_path = os.path.join(data_args.few_shot_eval_path, f"shots.{lg_pair}.json")
            if not os.path.isfile(pair_shot_path):
                ValueError(f"Make sure the language pair {lg_pair} is in the few shot eval folder!")
            with open(pair_shot_path) as f:
                shots_eval_dict[lg_pair] = json.load(f)

    train_datasets, eval_datasets, test_datasets = preprocess_cpo_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args)

    model = load_model(data_args, model_args, training_args, tokenizer, logger) 

    trainer = CPOTrainer(
        model,
        args=training_args,
        beta=model_args.cpo_beta,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        tokenizer=tokenizer,
        max_prompt_length=data_args.max_source_length,
        max_length=data_args.max_source_length+data_args.max_new_tokens,
        callbacks=[SavePeftModelCallback] if model_args.use_peft else None,
    )
    
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_state()
        if model_args.use_peft:
            if torch.distributed.get_rank() == 0:
                model.save_pretrained(training_args.output_dir) 
        else:
            trainer.save_model()

def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()