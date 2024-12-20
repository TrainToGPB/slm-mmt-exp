import yaml
import argparse


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_arguments_mbart(yaml_path):
    config = load_yaml_config(yaml_path)
    train_config = config['training']
    data_config = config['data']
    peft_config = config['peft']
    general_config = config['general']

    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument('--plm_name', type=str, default=train_config['plm_name'], help="Pretrained language model name (from HuggingFace)")
    parser.add_argument('--use_fsdp', type=lambda x: (str(x).lower() == 'true'), default=train_config['use_fsdp'], help="Use fully-sharded data parallel")
    parser.add_argument('--torch_dtype', type=str, default=train_config['torch_dtype'], help="Torch compute dtype: float16, bfloat16, float32")
    parser.add_argument('--seed', type=int, default=train_config['seed'], help="Random seed")
    parser.add_argument('--output_dir', type=str, default=train_config['output_dir'], help="Output directory")
    parser.add_argument('--src_lang', type=str, default=train_config['src_lang'], help="Source language")
    parser.add_argument('--tgt_lang', type=str, default=train_config['tgt_lang'], help="Target language")
    parser.add_argument('--max_length', type=int, default=train_config['max_length'], help="Max sequence length of encoder and decoder")
    parser.add_argument('--dataloader_num_workers', type=int, default=train_config['dataloader_num_workers'], help="Number of dataloader workers")
    parser.add_argument('--per_device_batch_size', type=int, default=train_config['per_device_batch_size'], help="Per device train/eval batch size")
    parser.add_argument('--optim', type=str, default=train_config['optim'], help="Optimizer type")
    parser.add_argument('--save_total_limit', type=int, default=train_config['save_total_limit'], help="Save total limit")
    parser.add_argument('--num_epochs', type=int, default=train_config['num_epochs'], help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=float(train_config['learning_rate']), help="Learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default=train_config['lr_scheduler_type'], help="LR scheduler type")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=train_config['gradient_accumulation_steps'], help="Gradient accumulation steps")
    parser.add_argument('--gradient_checkpointing', type=lambda x: (str(x).lower() == 'true'), default=train_config['gradient_checkpointing'], help="Use or not use gradient checkpointing")
    parser.add_argument('--max_grad_norm', type=float, default=train_config['max_grad_norm'], help="Maximum gradient norm")
    parser.add_argument('--warmup_ratio', type=float, default=train_config['warmup_ratio'], help="Warm-up ratio")
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help="Weight decay")
    parser.add_argument('--fp16', type=lambda x: (str(x).lower() == 'true'), default=train_config['fp16'], help="Use FP16")
    parser.add_argument('--bf16', type=lambda x: (str(x).lower() == 'true'), default=train_config['bf16'], help="Use BF16")
    parser.add_argument('--logging_dir', type=str, default=train_config['logging_dir'], help="Logging directory")
    parser.add_argument('--logging_strategy', type=str, default=train_config['logging_strategy'], help="Logging strategy")
    parser.add_argument('--evaluation_strategy', type=str, default=train_config['evaluation_strategy'], help="Evaluation strategy")
    parser.add_argument('--save_strategy', type=str, default=train_config['save_strategy'], help="Save strategy")
    parser.add_argument('--logging_steps', type=int, default=train_config['logging_steps'], help="Logging steps")
    parser.add_argument('--eval_steps', type=int, default=train_config['eval_steps'], help="Eval steps")
    parser.add_argument('--save_steps', type=int, default=train_config['save_steps'], help="Save steps")
    parser.add_argument('--report_to', type=str, default=train_config['report_to'], help="Report to")
    parser.add_argument('--load_best_model_at_end', type=lambda x: (str(x).lower() == 'true'), default=train_config['load_best_model_at_end'], help="Load best model at the end")
    parser.add_argument('--metric_for_best_model', type=str, default=train_config['metric_for_best_model'], help="Metric for best model")
    parser.add_argument('--just_test', type=lambda x: (str(x).lower() == 'true'), default=train_config['just_test'], help="데이터 적게, 스텝 짧게 테스트 용")

    # data config
    parser.add_argument('--dataset_name', type=str, default=data_config['dataset_name'], help="Dataset name (from HuggingFace)")

    # peft config
    parser.add_argument('--use_4bit', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_4bit'], help="Use 4-bit quantization")
    parser.add_argument('--use_8bit', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_8bit'], help="Use 8-bit quantization")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default=peft_config['bnb_4bit_quant_type'], help="BnB 4-bit quantization type")
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default=peft_config['bnb_4bit_compute_dtype'], help="BnB 4-bit compute dtype")
    parser.add_argument('--use_double_quant', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_double_quant'], help="Use double quantization")
    parser.add_argument('--lora_alpha', type=int, default=peft_config['lora_alpha'], help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=peft_config['lora_dropout'], help="LoRA dropout")
    parser.add_argument('--lora_r', type=int, default=peft_config['lora_r'], help="LoRA r")
    parser.add_argument('--lora_target_modules', type=lambda x: x.split(','), default=peft_config['lora_target_modules'], help="Modules where LoRA will adapt")

    # general config
    parser.add_argument('--project_name', type=str, default=general_config['project_name'], help="WandB project name")
    parser.add_argument('--run_name', type=str, default=general_config['run_name'], help="WandB run name")

    args = parser.parse_args()

    return args


def parse_arguments_llama(yaml_path):
    config = load_yaml_config(yaml_path)
    train_config = config['training']
    data_config = config['data']
    peft_config = config['peft']
    general_config = config['general']

    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument('--plm_name', type=str, default=train_config['plm_name'], help="Pretrained language model name (from HuggingFace)")
    parser.add_argument('--use_fsdp', type=lambda x: (str(x).lower() == 'true'), default=train_config['use_fsdp'], help="Use Fully-Sharded Data Parallel")
    parser.add_argument('--torch_dtype', type=str, default=train_config['torch_dtype'], help="Torch compute dtype: float16, bfloat16, float32")
    parser.add_argument('--output_dir', type=str, default=train_config['output_dir'], help="Output directory")
    parser.add_argument('--dataloader_num_workers', type=int, default=train_config['dataloader_num_workers'], help="Number of dataloader workers")
    parser.add_argument('--per_device_batch_size', type=int, default=train_config['per_device_batch_size'], help="Per device train/eval batch size")
    parser.add_argument('--group_by_length', type=lambda x: (str(x).lower() == 'true'), default=train_config['group_by_length'], help="Group by length")
    parser.add_argument('--max_length', type=int, default=train_config['max_length'], help="Maximum sequence length")
    parser.add_argument('--num_epochs', type=int, default=train_config['num_epochs'], help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=float(train_config['learning_rate']), help="Learning rate")
    parser.add_argument('--warmup_ratio', type=float, default=train_config['warmup_ratio'], help="Warm-up ratio")
    parser.add_argument('--lr_scheduler_type', type=str, default=train_config['lr_scheduler_type'], help="LR scheduler type")
    parser.add_argument('--lr_scheduler_num_cycles', type=int, default=train_config['lr_scheduler_num_cycles'], help="Number of cycles for 'cosine_with_restarts' scheduler")
    parser.add_argument('--optim', type=str, default=train_config['optim'], help="Optimizer type")
    parser.add_argument('--gradient_checkpointing', type=lambda x: (str(x).lower() == 'true'), default=train_config['gradient_checkpointing'], help="Gradient checkpointing")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=train_config['gradient_accumulation_steps'], help="Accumulation steps for gradient propagation")
    parser.add_argument('--max_grad_norm', type=float, default=train_config['max_grad_norm'], help="Maximum gradient norm")
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help="Weight decay")
    parser.add_argument('--packing', type=lambda x: (str(x).lower() == 'true'), default=train_config['packing'], help="Packing")
    parser.add_argument('--fp16', type=lambda x: (str(x).lower() == 'true'), default=train_config['fp16'], help="Use FP16")
    parser.add_argument('--bf16', type=lambda x: (str(x).lower() == 'true'), default=train_config['bf16'], help="Use BF16")
    parser.add_argument('--logging_dir', type=str, default=train_config['logging_dir'], help="Logging directory")
    parser.add_argument('--logging_strategy', type=str, default=train_config['logging_strategy'], help="Logging strategy")
    parser.add_argument('--evaluation_strategy', type=str, default=train_config['evaluation_strategy'], help="Evaluation strategy")
    parser.add_argument('--save_strategy', type=str, default=train_config['save_strategy'], help="Save strategy")
    parser.add_argument('--logging_steps', type=int, default=train_config['logging_steps'], help="Logging steps")
    parser.add_argument('--eval_steps', type=int, default=train_config['eval_steps'], help="Eval steps")
    parser.add_argument('--save_steps', type=int, default=train_config['save_steps'], help="Save steps")
    parser.add_argument('--save_total_limit', type=int, default=train_config['save_total_limit'], help="Save total limit")
    parser.add_argument('--seed', type=int, default=train_config['seed'], help="Random seed")
    parser.add_argument('--report_to', type=str, default=train_config['report_to'], help="Report to")
    parser.add_argument('--eval_accumulation_steps', type=int, default=train_config['eval_accumulation_steps'], help="Accumulations steps for evaluation")
    parser.add_argument('--eos_token_id', type=int, default=train_config['eos_token_id'], help="Token ID of end-of-sentence token")
    parser.add_argument('--pad_token_id', type=int, default=train_config['pad_token_id'], help="Token ID of padding token")
    parser.add_argument('--load_best_model_at_end', type=lambda x: (str(x).lower() == 'true'), default=train_config['load_best_model_at_end'], help="Load best model at the end")
    parser.add_argument('--metric_for_best_model', type=str, default=train_config['metric_for_best_model'], help="Metric for best model")
    parser.add_argument('--remove_unused_columns', type=lambda x: (str(x).lower() == 'true'), default=train_config['remove_unused_columns'], help="Remove dataset columns not used in training")
    parser.add_argument('--mix_word_dataset', type=lambda x: (str(x).lower() == 'true'), default=train_config['mix_word_dataset'], help="Mix word dataset")
    parser.add_argument('--just_test', action='store_true', help="데이터 적게, 스텝 짧게 테스트 용")

    # data config
    parser.add_argument('--train_dataset_name', type=str, default=data_config['train_dataset_name'], help="Train dataset name (from HuggingFace or local path)")
    parser.add_argument('--eval_dataset_name', type=str, default=data_config['eval_dataset_name'], help="Eval dataset name (from HuggingFace or local path)")
    parser.add_argument('--train_word_dataset_name', type=str, default=data_config['train_word_dataset_name'], help="Train word dataset name (from HuggingFace or local path)")
    parser.add_argument('--eval_word_dataset_name', type=str, default=data_config['eval_word_dataset_name'], help="Eval word dataset name (from HuggingFace or local path)")
    parser.add_argument('--train_inst_dataset_name', type=str, default=data_config['train_inst_dataset_name'], help="Train instruction dataset name (from HuggingFace or local path)")
    parser.add_argument('--eval_inst_dataset_name', type=str, default=data_config['eval_inst_dataset_name'], help="Eval instruction dataset name (from HuggingFace or local path)")
    parser.add_argument('--train_word_size', type=int, default=data_config['train_word_size'], help="Train word dataset size")
    parser.add_argument('--eval_word_size', type=int, default=data_config['eval_word_size'], help="Eval word dataset size")
    parser.add_argument('--train_inst_size', type=int, default=data_config['train_inst_size'], help="Train instruction dataset size")
    parser.add_argument('--eval_inst_size', type=int, default=data_config['eval_inst_size'], help="Eval instruction dataset size")
    parser.add_argument('--suffix_instruction', type=str, default=data_config['suffix_instruction'], help="Suffix for the instruction")
    parser.add_argument('--suffix_response', type=str, default=data_config['suffix_response'], help="Suffix for the prompt in target language")
    parser.add_argument('--lang_col1', type=str, default=data_config['lang_col1'], help="Column name for Language 1")
    parser.add_argument('--lang_col2', type=str, default=data_config['lang_col2'], help="Column name for Language 2")

    # peft config
    parser.add_argument('--use_4bit', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_4bit'], help="Use 4-bit quantization")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default=peft_config['bnb_4bit_quant_type'], help="BnB 4-bit quantization type")
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default=peft_config['bnb_4bit_compute_dtype'], help="BnB 4-bit compute dtype")
    parser.add_argument('--use_double_quant', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_double_quant'], help="Use double quantization")
    parser.add_argument('--use_lora', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_lora'], help="Use LoRA or not")
    parser.add_argument('--use_mora', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_mora'], help="Use MoRA or not (can be used only when LoRA is used)")
    parser.add_argument('--mora_type', type=int, default=peft_config['mora_type'], help="MoRA type")
    parser.add_argument('--lora_alpha', type=int, default=peft_config['lora_alpha'], help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=peft_config['lora_dropout'], help="LoRA dropout")
    parser.add_argument('--lora_r', type=int, default=peft_config['lora_r'], help="LoRA r")
    parser.add_argument('--lora_target_modules', type=lambda x: x.split(','), default=peft_config['lora_target_modules'], help="Modules where LoRA will adapt")
    parser.add_argument('--lora_target_layers', type=str, default=peft_config['lora_target_layers'], help="Layers where LoRA will adapt: all, odd, even")

    # general config
    parser.add_argument('--project_name', type=str, default=general_config['project_name'], help="WandB project name")
    parser.add_argument('--run_name', type=str, default=general_config['run_name'], help="WandB run name")

    args = parser.parse_args()

    return args


def parse_arguments_llama_dpo(yaml_path):
    config = load_yaml_config(yaml_path)

    train_config = config['training']
    data_config = config['data']
    peft_config = config['peft']
    dpo_config = config['dpo']
    general_config = config['general']

    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument('--plm_name', type=str, default=train_config['plm_name'], help="Pretrained language model name (from HuggingFace)")
    parser.add_argument('--use_fsdp', type=lambda x: (str(x).lower() == 'true'), default=train_config['use_fsdp'], help="Use Fully-Sharded Data Parallel")
    parser.add_argument('--torch_dtype', type=str, default=train_config['torch_dtype'], help="Torch compute dtype: float16, bfloat16, float32")
    parser.add_argument('--output_dir', type=str, default=train_config['output_dir'], help="Output directory")
    parser.add_argument('--dataloader_num_workers', type=int, default=train_config['dataloader_num_workers'], help="Number of dataloader workers")
    parser.add_argument('--per_device_batch_size', type=int, default=train_config['per_device_batch_size'], help="Per device train/eval batch size")
    parser.add_argument('--max_src_length', type=int, default=train_config['max_src_length'], help="Maximum source sequence length")
    parser.add_argument('--max_new_length', type=int, default=train_config['max_new_length'], help="Maximum new sequence length")
    parser.add_argument('--max_length', type=int, default=train_config['max_length'], help="Maximum sequence length")
    parser.add_argument('--num_epochs', type=int, default=train_config['num_epochs'], help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=float(train_config['learning_rate']), help="Learning rate")
    parser.add_argument('--warmup_ratio', type=float, default=train_config['warmup_ratio'], help="Warm-up ratio")
    parser.add_argument('--lr_scheduler_type', type=str, default=train_config['lr_scheduler_type'], help="LR scheduler type")
    parser.add_argument('--lr_scheduler_num_cycles', type=int, default=train_config['lr_scheduler_num_cycles'], help="Number of cycles for 'cosine_with_restarts' scheduler")
    parser.add_argument('--optim', type=str, default=train_config['optim'], help="Optimizer type")
    parser.add_argument('--gradient_checkpointing', type=lambda x: (str(x).lower() == 'true'), default=train_config['gradient_checkpointing'], help="Gradient checkpointing")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=train_config['gradient_accumulation_steps'], help="Accumulation steps for gradient propagation")
    parser.add_argument('--max_grad_norm', type=float, default=train_config['max_grad_norm'], help="Maximum gradient norm")
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help="Weight decay")
    parser.add_argument('--bf16', type=lambda x: (str(x).lower() == 'true'), default=train_config['bf16'], help="Use BF16")
    parser.add_argument('--logging_dir', type=str, default=train_config['logging_dir'], help="Logging directory")
    parser.add_argument('--logging_strategy', type=str, default=train_config['logging_strategy'], help="Logging strategy")
    parser.add_argument('--evaluation_strategy', type=str, default=train_config['evaluation_strategy'], help="Evaluation strategy")
    parser.add_argument('--save_strategy', type=str, default=train_config['save_strategy'], help="Save strategy")
    parser.add_argument('--logging_steps', type=int, default=train_config['logging_steps'], help="Logging steps")
    parser.add_argument('--eval_steps', type=int, default=train_config['eval_steps'], help="Eval steps")
    parser.add_argument('--save_steps', type=int, default=train_config['save_steps'], help="Save steps")
    parser.add_argument('--save_total_limit', type=int, default=train_config['save_total_limit'], help="Save total limit")
    parser.add_argument('--seed', type=int, default=train_config['seed'], help="Random seed")
    parser.add_argument('--report_to', type=str, default=train_config['report_to'], help="Report to")
    parser.add_argument('--eval_accumulation_steps', type=int, default=train_config['eval_accumulation_steps'], help="Accumulations steps for evaluation")
    parser.add_argument('--eos_token_id', type=int, default=train_config['eos_token_id'], help="Token ID of end-of-sentence token")
    parser.add_argument('--pad_token_id', type=int, default=train_config['pad_token_id'], help="Token ID of padding token")
    parser.add_argument('--load_best_model_at_end', type=lambda x: (str(x).lower() == 'true'), default=train_config['load_best_model_at_end'], help="Load best model at the end")
    parser.add_argument('--metric_for_best_model', type=str, default=train_config['metric_for_best_model'], help="Metric for best model")
    parser.add_argument('--remove_unused_columns', type=lambda x: (str(x).lower() == 'true'), default=train_config['remove_unused_columns'], help="Remove dataset columns not used in training")
    parser.add_argument('--just_test', action='store_true', help="데이터 적게, 스텝 짧게 테스트 용")

    # data config
    parser.add_argument('--train_dataset_name', type=str, default=data_config['train_dataset_name'], help="Train dataset name (from HuggingFace)")
    
    # peft config
    parser.add_argument('--use_4bit', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_4bit'], help="Use 4-bit quantization")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default=peft_config['bnb_4bit_quant_type'], help="BnB 4-bit quantization type")
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default=peft_config['bnb_4bit_compute_dtype'], help="BnB 4-bit compute dtype")
    parser.add_argument('--use_double_quant', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_double_quant'], help="Use double quantization")
    parser.add_argument('--use_lora', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_lora'], help="Use LoRA or not")
    parser.add_argument('--adapter_name', type=str, default=peft_config['adapter_name'], help="Adapter's name")

    # dpo config
    parser.add_argument('--dpo_loss_type', type=str, default=dpo_config['dpo_loss_type'], help="DPO loss type")
    parser.add_argument('--dpo_beta', type=float, default=dpo_config['dpo_beta'], help="DPO beta")
    parser.add_argument('--pol_adapter_name', type=str, default=dpo_config['pol_adapter_name'], help="POL adapter name")
    parser.add_argument('--ref_adapter_name', type=str, default=dpo_config['ref_adapter_name'], help="REF adapter name")

    # general config
    parser.add_argument('--project_name', type=str, default=general_config['project_name'], help="WandB project name")
    parser.add_argument('--run_name', type=str, default=general_config['run_name'], help="WandB run name")

    args = parser.parse_args()

    return args


def parse_arguments_mt5(yaml_path):
    config = load_yaml_config(yaml_path)
    train_config = config['training']
    data_config = config['data']
    general_config = config['general']

    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument('--seed', type=int, default=train_config['seed'], help="Random seed")
    parser.add_argument('--output_dir', type=str, default=train_config['output_dir'], help="Output directory")
    parser.add_argument('--plm_name', type=str, default=train_config['plm_name'], help="Pretrained language model name (from HuggingFace)")
    parser.add_argument('--lang_list', type=str, default=train_config['lang_list'], help="List of languages")
    parser.add_argument('--use_fsdp', type=lambda x: (str(x).lower() == 'true'), default=train_config['use_fsdp'], help="Use Fully-Sharded Data Parallel")
    parser.add_argument('--max_length', type=int, default=train_config['max_length'], help="Max sequence length of encoder and decoder")
    parser.add_argument('--dataloader_num_workers', type=int, default=train_config['dataloader_num_workers'], help="Number of dataloader workers")
    parser.add_argument('--per_device_train_batch_size', type=int, default=train_config['per_device_train_batch_size'], help="Per device train batch size")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=train_config['per_device_eval_batch_size'], help="Per device eval batch size")
    parser.add_argument('--save_total_limit', type=int, default=train_config['save_total_limit'], help="Save total limit")
    parser.add_argument('--num_epochs', type=int, default=train_config['num_epochs'], help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=float(train_config['learning_rate']), help="Learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default=train_config['lr_scheduler_type'], help="LR scheduler type")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=train_config['gradient_accumulation_steps'], help="Gradient accumulation steps")
    parser.add_argument('--gradient_checkpointing', type=lambda x: (str(x).lower() == 'true'), default=train_config['gradient_checkpointing'], help="Use or not use gradient checkpointing")
    parser.add_argument('--warmup_ratio', type=float, default=train_config['warmup_ratio'], help="Warm-up ratio")
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help="Weight decay")
    parser.add_argument('--fp16', type=lambda x: (str(x).lower() == 'true'), default=train_config['fp16'], help="Use FloatingPoint 16-bit to compute")
    parser.add_argument('--logging_dir', type=str, default=train_config['logging_dir'], help="Logging directory")
    parser.add_argument('--logging_strategy', type=str, default=train_config['logging_strategy'], help="Logging strategy")
    parser.add_argument('--evaluation_strategy', type=str, default=train_config['evaluation_strategy'], help="Evaluation strategy")
    parser.add_argument('--save_strategy', type=str, default=train_config['save_strategy'], help="Save strategy")
    parser.add_argument('--logging_steps', type=int, default=train_config['logging_steps'], help="Logging steps")
    parser.add_argument('--eval_steps', type=int, default=train_config['eval_steps'], help="Eval steps")
    parser.add_argument('--save_steps', type=int, default=train_config['save_steps'], help="Save steps")
    parser.add_argument('--report_to', type=str, default=train_config['report_to'], help="Report to")
    parser.add_argument('--load_best_model_at_end', type=lambda x: (str(x).lower() == 'true'), default=train_config['load_best_model_at_end'], help="Load best model at the end")
    parser.add_argument('--metric_for_best_model', type=str, default=train_config['metric_for_best_model'], help="Metric for best model")
    parser.add_argument('--include_inputs_for_metrics', type=str, default=train_config['include_inputs_for_metrics'], help="Whether to include inputs for metrics")
    parser.add_argument('--translation_mode', type=str, default=train_config['translation_mode'], help="Translation mode")
    parser.add_argument('--just_test', type=lambda x: (str(x).lower() == 'true'), default=train_config['just_test'], help="데이터 적게, 스텝 짧게 테스트 용")

    # data config
    parser.add_argument('--dataset_name', type=str, default=data_config['dataset_name'], help="Dataset name (from HuggingFace)")

    # general config
    parser.add_argument('--project_name', type=str, default=general_config['project_name'], help="WandB project name")
    parser.add_argument('--run_name', type=str, default=general_config['run_name'], help="WandB run name")

    args = parser.parse_args()

    return args


def parse_arguments_mt5_qlora(yaml_path):
    config = load_yaml_config(yaml_path)
    train_config = config['training']
    data_config = config['data']
    peft_config = config['peft']
    general_config = config['general']

    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument('--seed', type=int, default=train_config['seed'], help="Random seed")
    parser.add_argument('--output_dir', type=str, default=train_config['output_dir'], help="Output directory")
    parser.add_argument('--plm_name', type=str, default=train_config['plm_name'], help="Pretrained language model name (from HuggingFace)")
    parser.add_argument('--lang_list', type=str, default=train_config['lang_list'], help="List of languages")
    parser.add_argument('--use_fsdp', type=lambda x: (str(x).lower() == 'true'), default=train_config['use_fsdp'], help="Use Fully-Sharded Data Parallel")
    parser.add_argument('--device_map', type=str, default=train_config['device_map'], help="Device where to model put on")
    parser.add_argument('--max_length', type=int, default=train_config['max_length'], help="Max sequence length of encoder and decoder")
    parser.add_argument('--dataloader_num_workers', type=int, default=train_config['dataloader_num_workers'], help="Number of dataloader workers")
    parser.add_argument('--per_device_train_batch_size', type=int, default=train_config['per_device_train_batch_size'], help="Per device train batch size")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=train_config['per_device_eval_batch_size'], help="Per device eval batch size")
    parser.add_argument('--save_total_limit', type=int, default=train_config['save_total_limit'], help="Save total limit")
    parser.add_argument('--num_epochs', type=int, default=train_config['num_epochs'], help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=float(train_config['learning_rate']), help="Learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default=train_config['lr_scheduler_type'], help="LR scheduler type")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=train_config['gradient_accumulation_steps'], help="Gradient accumulation steps")
    parser.add_argument('--gradient_checkpointing', type=lambda x: (str(x).lower() == 'true'), default=train_config['gradient_checkpointing'], help="Use or not use gradient checkpointing")
    parser.add_argument('--max_grad_norm', type=float, default=train_config['max_grad_norm'], help="Maximum gradient norm")
    parser.add_argument('--warmup_ratio', type=float, default=train_config['warmup_ratio'], help="Warm-up ratio")
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help="Weight decay")
    parser.add_argument('--fp16', type=lambda x: (str(x).lower() == 'true'), default=train_config['fp16'], help="Use FloatingPoint 16-bit to compute")
    parser.add_argument('--bf16', type=lambda x: (str(x).lower() == 'true'), default=train_config['bf16'], help="Use BrainFloat 16-bit to compute")
    parser.add_argument('--logging_dir', type=str, default=train_config['logging_dir'], help="Logging directory")
    parser.add_argument('--logging_strategy', type=str, default=train_config['logging_strategy'], help="Logging strategy")
    parser.add_argument('--evaluation_strategy', type=str, default=train_config['evaluation_strategy'], help="Evaluation strategy")
    parser.add_argument('--save_strategy', type=str, default=train_config['save_strategy'], help="Save strategy")
    parser.add_argument('--logging_steps', type=int, default=train_config['logging_steps'], help="Logging steps")
    parser.add_argument('--eval_steps', type=int, default=train_config['eval_steps'], help="Eval steps")
    parser.add_argument('--save_steps', type=int, default=train_config['save_steps'], help="Save steps")
    parser.add_argument('--report_to', type=str, default=train_config['report_to'], help="Report to")
    parser.add_argument('--load_best_model_at_end', type=lambda x: (str(x).lower() == 'true'), default=train_config['load_best_model_at_end'], help="Load best model at the end")
    parser.add_argument('--metric_for_best_model', type=str, default=train_config['metric_for_best_model'], help="Metric for best model")
    parser.add_argument('--include_inputs_for_metrics', type=str, default=train_config['include_inputs_for_metrics'], help="Whether to include inputs for metrics")
    parser.add_argument('--translation_mode', type=str, default=train_config['translation_mode'], help="Translation mode")
    parser.add_argument('--just_test', type=lambda x: (str(x).lower() == 'true'), default=train_config['just_test'], help="데이터 적게, 스텝 짧게 테스트 용")

    # data config
    parser.add_argument('--dataset_name', type=str, default=data_config['dataset_name'], help="Dataset name (from HuggingFace)")

    # peft config
    parser.add_argument('--use_4bit', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_4bit'], help="Use 4-bit quantization")
    parser.add_argument('--use_8bit', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_8bit'], help="Use 8-bit quantization")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default=peft_config['bnb_4bit_quant_type'], help="BnB 4-bit quantization type")
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default=peft_config['bnb_4bit_compute_dtype'], help="BnB 4-bit compute dtype")
    parser.add_argument('--use_double_quant', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_double_quant'], help="Use double quantization")
    parser.add_argument('--optim', type=str, default=peft_config['optim'], help="Optimizer type")
    parser.add_argument('--lora_alpha', type=int, default=peft_config['lora_alpha'], help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=peft_config['lora_dropout'], help="LoRA dropout")
    parser.add_argument('--lora_r', type=int, default=peft_config['lora_r'], help="LoRA r")
    parser.add_argument('--lora_target_modules', type=lambda x: x.split(','), default=peft_config['lora_target_modules'], help="Modules where LoRA will adapt")
    parser.add_argument('--lora_path', type=str, default=peft_config['lora_path'], help="Trained LoRA adapter's path")

    # general config
    parser.add_argument('--project_name', type=str, default=general_config['project_name'], help="WandB project name")
    parser.add_argument('--run_name', type=str, default=general_config['run_name'], help="WandB run name")

    args = parser.parse_args()

    return args


def parse_arguments_madlad_qlora(yaml_path):
    config = load_yaml_config(yaml_path)
    train_config = config['training']
    data_config = config['data']
    peft_config = config['peft']
    general_config = config['general']

    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument('--seed', type=int, default=train_config['seed'], help="Random seed")
    parser.add_argument('--output_dir', type=str, default=train_config['output_dir'], help="Output directory")
    parser.add_argument('--plm_name', type=str, default=train_config['plm_name'], help="Pretrained language model name (from HuggingFace)")
    parser.add_argument('--lang_list', type=str, default=train_config['lang_list'], help="List of languages")
    parser.add_argument('--use_fsdp', type=lambda x: (str(x).lower() == 'true'), default=train_config['use_fsdp'], help="Use Fully-Sharded Data Parallel")
    parser.add_argument('--device_map', type=str, default=train_config['device_map'], help="Device where to model put on")
    parser.add_argument('--max_length', type=int, default=train_config['max_length'], help="Max sequence length of encoder and decoder")
    parser.add_argument('--dataloader_num_workers', type=int, default=train_config['dataloader_num_workers'], help="Number of dataloader workers")
    parser.add_argument('--per_device_train_batch_size', type=int, default=train_config['per_device_train_batch_size'], help="Per device train batch size")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=train_config['per_device_eval_batch_size'], help="Per device eval batch size")
    parser.add_argument('--save_total_limit', type=int, default=train_config['save_total_limit'], help="Save total limit")
    parser.add_argument('--num_epochs', type=int, default=train_config['num_epochs'], help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=float(train_config['learning_rate']), help="Learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default=train_config['lr_scheduler_type'], help="LR scheduler type")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=train_config['gradient_accumulation_steps'], help="Gradient accumulation steps")
    parser.add_argument('--gradient_checkpointing', type=lambda x: (str(x).lower() == 'true'), default=train_config['gradient_checkpointing'], help="Use or not use gradient checkpointing")
    parser.add_argument('--max_grad_norm', type=float, default=train_config['max_grad_norm'], help="Maximum gradient norm")
    parser.add_argument('--warmup_ratio', type=float, default=train_config['warmup_ratio'], help="Warm-up ratio")
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help="Weight decay")
    parser.add_argument('--fp16', type=lambda x: (str(x).lower() == 'true'), default=train_config['fp16'], help="Use FloatingPoint 16-bit to compute")
    parser.add_argument('--bf16', type=lambda x: (str(x).lower() == 'true'), default=train_config['bf16'], help="Use BrainFloat 16-bit to compute")
    parser.add_argument('--logging_dir', type=str, default=train_config['logging_dir'], help="Logging directory")
    parser.add_argument('--logging_strategy', type=str, default=train_config['logging_strategy'], help="Logging strategy")
    parser.add_argument('--evaluation_strategy', type=str, default=train_config['evaluation_strategy'], help="Evaluation strategy")
    parser.add_argument('--save_strategy', type=str, default=train_config['save_strategy'], help="Save strategy")
    parser.add_argument('--logging_steps', type=int, default=train_config['logging_steps'], help="Logging steps")
    parser.add_argument('--eval_steps', type=int, default=train_config['eval_steps'], help="Eval steps")
    parser.add_argument('--save_steps', type=int, default=train_config['save_steps'], help="Save steps")
    parser.add_argument('--report_to', type=str, default=train_config['report_to'], help="Report to")
    parser.add_argument('--load_best_model_at_end', type=lambda x: (str(x).lower() == 'true'), default=train_config['load_best_model_at_end'], help="Load best model at the end")
    parser.add_argument('--metric_for_best_model', type=str, default=train_config['metric_for_best_model'], help="Metric for best model")
    parser.add_argument('--include_inputs_for_metrics', type=str, default=train_config['include_inputs_for_metrics'], help="Whether to include inputs for metrics")
    parser.add_argument('--translation_mode', type=str, default=train_config['translation_mode'], help="Translation mode")
    parser.add_argument('--just_test', type=lambda x: (str(x).lower() == 'true'), default=train_config['just_test'], help="데이터 적게, 스텝 짧게 테스트 용")

    # data config
    parser.add_argument('--dataset_name', type=str, default=data_config['dataset_name'], help="Dataset name (from HuggingFace)")

    # peft config
    parser.add_argument('--use_4bit', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_4bit'], help="Use 4-bit quantization")
    parser.add_argument('--use_8bit', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_8bit'], help="Use 8-bit quantization")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default=peft_config['bnb_4bit_quant_type'], help="BnB 4-bit quantization type")
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default=peft_config['bnb_4bit_compute_dtype'], help="BnB 4-bit compute dtype")
    parser.add_argument('--use_double_quant', type=lambda x: (str(x).lower() == 'true'), default=peft_config['use_double_quant'], help="Use double quantization")
    parser.add_argument('--optim', type=str, default=peft_config['optim'], help="Optimizer type")
    parser.add_argument('--lora_alpha', type=int, default=peft_config['lora_alpha'], help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=peft_config['lora_dropout'], help="LoRA dropout")
    parser.add_argument('--lora_r', type=int, default=peft_config['lora_r'], help="LoRA r")
    parser.add_argument('--lora_target_modules', type=lambda x: x.split(','), default=peft_config['lora_target_modules'], help="Modules where LoRA will adapt")
    parser.add_argument('--lora_path', type=str, default=peft_config['lora_path'], help="Trained LoRA adapter's path")

    # general config
    parser.add_argument('--project_name', type=str, default=general_config['project_name'], help="WandB project name")
    parser.add_argument('--run_name', type=str, default=general_config['run_name'], help="WandB run name")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    yaml_path = '../ojt_translation/training/mbart/codes/mbart_config.yaml'
    args = parse_arguments_mbart(yaml_path)
    print(args)