# built-in
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys

# third-party
import torch
import wandb
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
from datasets import Dataset
from accelerate import Accelerator
from transformers import BitsAndBytesConfig

# custom
sys.path.append(os.path.join(SCRIPT_DIR, '../../'))
from training_utils import set_seed
from argument import load_yaml_config, parse_arguments_llama_dpo
from dpo_utils import SavePeftModelCallback
sys.path.append(os.path.join(SCRIPT_DIR, '../../../../'))
try:
    from custom_utils.general_secret import WANDB_CLIENT_KEY # Use your own path
except:
    WANDB_CLIENT_KEY = "YOUR_WANDB_CLIENT_KEY"


STORE_PATH = "/data/sehyeong/nmt"
LANG_TABLE = {
    "ko": "한국어",
    "en": "English",
    "ja": "日本語",
    "zh": "中文",
}
ADAPTER_MAPPING = {
    'mmt-v1': os.path.join(STORE_PATH, 'models/mmt-ft/ko-enjazh/v1-it-sft-xml'),
}
HF_MODEL_CACHE_DIR = "/data/.cache/hub/"
HF_DATASETS_CACHE_DIR = "/data/.cache/datasets/"


def load_model_and_tokenizer(args):
    # Model (quantization, adapter)
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

    adapter_path = ADAPTER_MAPPING[args.adapter_name]

    # Policy adapter (trainable)
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=True,
        adapter_name=args.pol_adapter_name,
    )
    model.print_trainable_parameters()

    # Reference adapter (frozen)
    model.load_adapter(
        adapter_path,
        is_trainable=False,
        adapter_name=args.ref_adapter_name,
    )
    
    # Tokenizer
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

    return model, tokenizer


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
    steps_per_epoch = (dataset_size / batch_size)
    total_steps = epochs * steps_per_epoch / gradient_accumulation_steps
    total_steps_per_device = total_steps / torch.cuda.device_count()
    warmup_steps = int(total_steps_per_device * warmup_ratio)
    warmup_steps = warmup_steps if warmup_steps > 1 else 0
    return warmup_steps


def preprocess_dataset(dataset):
    def get_prompt(example):
        src_lang_code, tgt_lang_code = example['direction'].split('-')
        src_lang, tgt_lang = LANG_TABLE[src_lang_code], LANG_TABLE[tgt_lang_code]
        src_text = example['src']
        guides = eval(example['guideline'])

        instruction_part = {
            'head': "<instruction>",
            'body': f"Translate the source sentence from {src_lang} to {tgt_lang}.\nBe sure to reflect the guidelines below when translating.",
            'tail': "</instruction>"
        }
        instruction = f"{instruction_part['head']}\n{instruction_part['body']}\n{instruction_part['tail']}"
        guideline_part = {
            'head': "<guideline>",
            'body': guides,
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
        }
        tgt = f"{tgt_part['head']}\n"
        translation_part = {
            'head': "<translation>",
            'body': f"{src}\n{tgt}",
        }
        translation = f"{translation_part['head']}\n{translation_part['body']}"
        
        prompt = f"{instruction}\n\n{guideline}\n\n{translation}"

        return prompt
    
    def get_chosen_rejected(example):
        tgt_lang_code = example['direction'].split('-')[1]
        tgt_lang = LANG_TABLE[tgt_lang_code]
        chosen = '\n'.join([example['tgt-chosen'], f'</{tgt_lang}></target>'])
        rejected = '\n'.join([example['tgt-rejected'], f'</{tgt_lang}></target>'])
        return chosen, rejected
    
    dataset = pd.DataFrame(dataset)
    prompts = dataset.apply(get_prompt, axis=1)
    chosens_rejecteds = dataset.apply(get_chosen_rejected, axis=1)
    chosens, rejecteds = zip(*chosens_rejecteds)
    processed_examples = {
        'prompt': prompts,
        'chosen': chosens,
        'rejected': rejecteds,
    }

    processed_df = pd.DataFrame(processed_examples)
    processed_dataset = Dataset.from_pandas(processed_df)

    return processed_dataset


def train(args):
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)

    train_dataset = load_dataset(args.train_dataset_name, cache_dir=HF_DATASETS_CACHE_DIR)['train']
    if args.just_test:
        train_dataset = train_dataset.select(range(1000))
    train_dataset = preprocess_dataset(train_dataset)
    print(train_dataset[0])

    dataset_size = len(train_dataset)
    warmup_steps = calculate_warmup_steps(
        epochs=args.num_epochs,
        dataset_size=dataset_size,
        batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio
    )
    print("WARMUP STEPS:", warmup_steps)

    if args.just_test:
        args.project_name = 'test'
        args.output_dir = os.path.join(STORE_PATH, 'models/test')
        args.logging_steps = 1
        args.save_steps = 1
        train_dataset = train_dataset.select(range(1000))
    
    args.output_dir = os.path.join(STORE_PATH, args.output_dir)

    wandb.login(
        anonymous='never',
        key=WANDB_CLIENT_KEY,
        relogin=True,
        force=True
    )
    wandb.init(project=args.project_name, name=args.run_name)

    training_args = DPOConfig(
        loss_type=args.dpo_loss_type,
        output_dir=args.output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps, # 전체 step의 10%
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        fp16=False,
        bf16=args.bf16,
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps, # 전체 step의 10%
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        max_length=args.max_length,
        max_prompt_length=args.max_src_length,
        max_target_length=args.max_new_length,
        model_adapter_name=args.pol_adapter_name,
        ref_adapter_name=args.ref_adapter_name,
        callbacks=[SavePeftModelCallback],
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == '__main__':
    yaml_path = '../configs/llama_dpo_config.yaml'
    args = parse_arguments_llama_dpo(yaml_path)
    acc_yaml_path = '../configs/deepspeed_train_config_bf16.yaml'
    acc_config = load_yaml_config(acc_yaml_path)

    args.gradient_accumulation_steps = acc_config['deepspeed_config']['gradient_accumulation_steps']
    args.max_grad_norm = acc_config['deepspeed_config']['gradient_clipping']
    args.dataloader_num_workers = acc_config['num_processes']

    train(args)
