# built-in
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# third-party
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import DPOTrainer
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig

# custom
sys.path.append('../../')
from training_utils import set_seed
from argument import load_yaml_config, parse_arguments_llama_dpo
from dpo_utils import SavePeftModelCallback, preprocess_cpo_data
sys.path.append('../../../../') # Use your own path
from custom_utils.general_secret import WANDB_CLIENT_KEY # Use your own path


def load_model_and_tokenizer(plm_name, lora_path, device_map, bnb_config, pol_adapter_name, ref_adapter_name):
    model = AutoModelForCausalLM.from_pretrained(
        plm_name,
        quantization_config=bnb_config,
        torch_dtype=bnb_config.bnb_4bit_compute_dtype,
        attn_implementation='flash_attention_2',
        device_map=device_map
    )
    model.config.use_cache = False

    # Policy adapter (trainable)
    model = PeftModel.from_pretrained(
        model,
        model_id=lora_path,
        is_trainable=True,
        adapter_name=pol_adapter_name,
    )
    model.print_trainable_parameters()

    # Reference adapter (frozen)
    model.load_adapter(
        lora_path,
        adapter_name=ref_adapter_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(plm_name, trust_remote_code=True)

    return model, tokenizer


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
    steps_per_epoch = (dataset_size / batch_size)
    total_steps = epochs * steps_per_epoch / gradient_accumulation_steps
    total_steps_per_device = total_steps / torch.cuda.device_count()
    warmup_steps = int(total_steps_per_device * warmup_ratio)
    return warmup_steps


def train(args):
    set_seed(args.seed)

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_double_quant
    )
    model, tokenizer = load_model_and_tokenizer(
        plm_name=args.plm_name, 
        lora_path=args.lora_path,
        # device_map=args.device_map, 
        device_map=torch.cuda.current_device(),
        bnb_config=bnb_config, 
        pol_adapter_name=args.policy_adapter,
        ref_adapter_name=args.reference_adapter,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=None, # 전체 step의 10%
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
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps, # 전체 step의 10%
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
    )

    dataset = preprocess_cpo_data(
        args.dataset_name, 
        args.language_pairs, 
        tokenizer, 
        args, 
        training_args,
    )

    dataset_size = len(dataset)
    warmup_steps = calculate_warmup_steps(
        epochs=args.num_epochs,
        dataset_size=dataset_size,
        batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio
    )
    training_args.warmup_steps = warmup_steps

    if args.just_test:
        training_args.output_dir = '../models/test'
        training_args.logging_steps = 1
        training_args.eval_steps = 1
        training_args.save_steps = 1
        project_name = 'test'
        train_dataset = dataset.select(range(300))
    else:
        project_name = args.project_name
        train_dataset = dataset
    
    wandb.login(
        anonymous='never',
        key=WANDB_CLIENT_KEY,
        relogin=True,
        force=True
    )
    wandb.init(project=project_name, name=args.run_name)

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        max_length=args.max_src_length + args.max_new_length,
        max_prompt_length=args.max_src_length,
        max_target_length=args.max_new_length,
        model_adapter_name=args.policy_adapter,
        ref_adapter_name=args.reference_adapter,
        callbacks=[SavePeftModelCallback],
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    yaml_path = '../configs/llama_dpo_config.yaml'
    args = parse_arguments_llama_dpo(yaml_path)
    acc_yaml_path = '../configs/deepspeed_train_config_bf16.yaml'
    acc_config = load_yaml_config(acc_yaml_path)

    args.gradient_accumulation_steps = acc_config['deepspeed_config']['gradient_accumulation_steps']
    args.max_grad_norm = acc_config['deepspeed_config']['gradient_clipping']
    args.dataloader_num_workers = acc_config['num_processes']

    train(args)
