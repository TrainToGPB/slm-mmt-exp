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
    qlora_config = config['qlora']
    general_config = config['general']

    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument('--seed', type=int, default=train_config['seed'], help="Random seed")
    parser.add_argument('--output_dir', type=str, default=train_config['output_dir'], help="Output directory")
    parser.add_argument('--plm_name', type=str, default=train_config['plm_name'], help="Pretrained language model name (from HuggingFace)")
    parser.add_argument('--src_lang', type=str, default=train_config['src_lang'], help="Source language")
    parser.add_argument('--tgt_lang', type=str, default=train_config['tgt_lang'], help="Target language")
    parser.add_argument('--use_fsdp', type=lambda x: (str(x).lower() == 'true'), default=train_config['use_fsdp'], help="Use Fully-Sharded Data Parallel")
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

    # qlora config
    parser.add_argument('--use_4bit', type=lambda x: (str(x).lower() == 'true'), default=qlora_config['use_4bit'], help="Use 4-bit quantization")
    parser.add_argument('--use_8bit', type=lambda x: (str(x).lower() == 'true'), default=qlora_config['use_8bit'], help="Use 8-bit quantization")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default=qlora_config['bnb_4bit_quant_type'], help="BnB 4-bit quantization type")
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default=qlora_config['bnb_4bit_compute_dtype'], help="BnB 4-bit compute dtype")
    parser.add_argument('--use_double_quant', type=lambda x: (str(x).lower() == 'true'), default=qlora_config['use_double_quant'], help="Use double quantization")
    parser.add_argument('--lora_alpha', type=int, default=qlora_config['lora_alpha'], help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=qlora_config['lora_dropout'], help="LoRA dropout")
    parser.add_argument('--lora_r', type=int, default=qlora_config['lora_r'], help="LoRA r")
    parser.add_argument('--lora_target_modules', type=lambda x: x.split(','), default=qlora_config['lora_target_modules'], help="Modules where LoRA will adapt")

    # general config
    parser.add_argument('--project_name', type=str, default=general_config['project_name'], help="WandB project name")
    parser.add_argument('--run_name', type=str, default=general_config['run_name'], help="WandB run name")

    args = parser.parse_args()

    return args


def parse_arguments_llama(yaml_path):
    config = load_yaml_config(yaml_path)
    train_config = config['training']
    data_config = config['data']
    qlora_config = config['qlora']
    general_config = config['general']

    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument('--plm_name', type=str, default=train_config['plm_name'], help="Pretrained language model name (from HuggingFace)")
    parser.add_argument('--use_fsdp', type=lambda x: (str(x).lower() == 'true'), default=train_config['use_fsdp'], help="Use Fully-Sharded Data Parallel")
    parser.add_argument('--device_map', type=bool, default=train_config['device_map'], help="Device where to model put on")
    parser.add_argument('--output_dir', type=str, default=train_config['output_dir'], help="Output directory")
    parser.add_argument('--dataloader_num_workers', type=int, default=train_config['dataloader_num_workers'], help="Number of dataloader workers")
    parser.add_argument('--per_device_batch_size', type=int, default=train_config['per_device_batch_size'], help="Per device train/eval batch size")
    parser.add_argument('--group_by_length', type=lambda x: (str(x).lower() == 'true'), default=train_config['group_by_length'], help="Group by length")
    parser.add_argument('--max_length', type=int, default=train_config['max_length'], help="Maximum sequence length")
    parser.add_argument('--num_epochs', type=int, default=train_config['num_epochs'], help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=float(train_config['learning_rate']), help="Learning rate")
    parser.add_argument('--warmup_ratio', type=float, default=train_config['warmup_ratio'], help="Warm-up ratio")
    parser.add_argument('--lr_scheduler_type', type=str, default=train_config['lr_scheduler_type'], help="LR scheduler type")
    parser.add_argument('--optim', type=str, default=train_config['optim'], help="Optimizer type")
    parser.add_argument('--gradient_checkpointing', type=lambda x: (str(x).lower() == 'true'), default=train_config['gradient_checkpointing'], help="Gradient checkpointing")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=train_config['gradient_accumulation_steps'], help="Accumulation steps for gradient propagation")
    parser.add_argument('--max_grad_norm', type=float, default=train_config['max_grad_norm'], help="Maximum gradient norm")
    parser.add_argument('--weight_decay', type=float, default=train_config['weight_decay'], help="Weight decay")
    parser.add_argument('--packing', type=bool, default=train_config['packing'], help="Packing")
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
    parser.add_argument('--load_best_model_at_end', type=lambda x: (str(x).lower() == 'true'), default=train_config['load_best_model_at_end'], help="Load best model at the end")
    parser.add_argument('--metric_for_best_model', type=str, default=train_config['metric_for_best_model'], help="Metric for best model")
    parser.add_argument('--remove_unused_columns', type=lambda x: (str(x).lower() == 'true'), default=train_config['remove_unused_columns'], help="Remove dataset columns not used in training")
    parser.add_argument('--just_test', type=lambda x: (str(x).lower() == 'true'), default=train_config['just_test'], help="데이터 적게, 스텝 짧게 테스트 용")

    # data config
    parser.add_argument('--dataset_name', type=str, default=data_config['dataset_name'], help="Dataset name (from HuggingFace)")
    parser.add_argument('--instruction', type=str, default=data_config['instruction'], help="Training instruction for the model.")
    parser.add_argument('--en_sign', type=str, default=data_config['en_sign'], help="Instruction or prompt that the following sentence is English.")
    parser.add_argument('--ko_sign', type=str, default=data_config['ko_sign'], help="Instruction or prompt that the following sentence is Korean.")

    # qlora config
    parser.add_argument('--use_4bit', type=lambda x: (str(x).lower() == 'true'), default=qlora_config['use_4bit'], help="Use 4-bit quantization")
    parser.add_argument('--use_8bit', type=lambda x: (str(x).lower() == 'true'), default=qlora_config['use_8bit'], help="Use 8-bit quantization")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default=qlora_config['bnb_4bit_quant_type'], help="BnB 4-bit quantization type")
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default=qlora_config['bnb_4bit_compute_dtype'], help="BnB 4-bit compute dtype")
    parser.add_argument('--use_double_quant', type=lambda x: (str(x).lower() == 'true'), default=qlora_config['use_double_quant'], help="Use double quantization")
    parser.add_argument('--lora_alpha', type=int, default=qlora_config['lora_alpha'], help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=qlora_config['lora_dropout'], help="LoRA dropout")
    parser.add_argument('--lora_r', type=int, default=qlora_config['lora_r'], help="LoRA r")
    parser.add_argument('--lora_target_modules', type=lambda x: x.split(','), default=qlora_config['lora_target_modules'], help="Modules where LoRA will adapt")
    parser.add_argument('--lora_path', type=str, default=qlora_config['lora_path'], help="Trained LoRA adapter's path")

    # general config
    parser.add_argument('--project_name', type=str, default=general_config['project_name'], help="WandB project name")
    parser.add_argument('--run_name', type=str, default=general_config['run_name'], help="WandB run name")

    args = parser.parse_args()

    return args




if __name__ == '__main__':
    yaml_path = '../ojt_translation/training/mbart/codes/mbart_config.yaml'
    args = parse_arguments_mbart(yaml_path)
    print(args)