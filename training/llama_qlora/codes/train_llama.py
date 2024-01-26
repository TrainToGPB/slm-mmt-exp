# built-in
import os
import sys
import random
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# third-party
import torch
import numpy as np
import wandb
import evaluate
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BitsAndBytesConfig
from transformers import default_data_collator
import bitsandbytes as bnb
import torch.distributed.checkpoint as DCP

# custom
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
from wandb_secret import WANDB_CLIENT_KEY
from argument import parse_arguments_llama


def set_seed(SEED=42):
    """
    Set the random seeds for reproducibility in a PyTorch environment.

    Parameters:
    - SEED (int, optional): Seed value to be used for random number generation. Default is 42.

    Usage:
    Call this function before running any code that involves random number generation to ensure reproducibility.

    Example:
    set_seed(123)
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def load_model_and_tokenizer(
        plm_name,
        use_4bit,
        bnb_4bit_quant_type,
        bnb_4bit_compute_dtype,
        use_double_quant,
        device_map
    ):
    """
    Load a pre-trained language model and tokenizer with optional bits-and-bytes quantization.

    Parameters:
    - plm_name (str): Pre-trained language model name.
    - use_4bit (bool): Whether to use 4-bit quantization.
    - bnb_4bit_quant_type (str): Quantization type for 4-bit quantization.
    - bnb_4bit_compute_dtype (str): Data type for computation during 4-bit quantization.
    - use_double_quant (bool): Whether to use double quantization.
    - device_map (int): CUDA device ID.

    Returns:
    - model (PreTrainedModel): Loaded pre-trained language model.
    - tokenizer (PreTrainedTokenizer): Loaded tokenizer.
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant
    )

    model = AutoModelForCausalLM.from_pretrained(
        plm_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map=torch.cuda.current_device()
    )
    model.config.use_cache = False
    model.config.pretrining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(plm_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return model, tokenizer


def preprocess_example(example, en_sign, ko_sign, max_length, tokenizer):
    """
    Preprocess a single example for training or evaluation.

    Parameters:
    - example (dict): Single example containing English and Korean text.
    - en_sign (str): English signature.
    - ko_sign (str): Korean signature.
    - max_length (int): Maximum length of the input sequence.
    - tokenizer (PreTrainedTokenizer): Tokenizer for encoding the input.

    Returns:
    - model_inputs (dict): Model inputs for the preprocessed example.
    """
    prompt = " ".join([en_sign, example['en'], ko_sign])
    response = example['ko']

    model_inputs = tokenizer(
        prompt, 
        text_target=response, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True,
        return_tensors="pt"
    )
    model_inputs = {key: value.squeeze() for key, value in model_inputs.items()}

    return model_inputs


def postprocess_text(preds, labels):
    """
    Postprocess model predictions and labels.

    Parameters:
    - preds (list): Model predictions.
    - labels (list): Ground truth labels.

    Returns:
    - tuple: Tuple containing postprocessed predictions and labels.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def find_all_linear_names(model, use_4bit=True, use_8bit=False):
    """
    Find all linear module names in the LoRA-adapted model.

    Parameters:
    - model (PreTrainedModel): LoRA-adapted model.
    - use_4bit (bool): Whether to use 4-bit quantization.
    - use_8bit (bool): Whether to use 8-bit quantization.

    Returns:
    - list: List of linear module names.
    """
    if use_4bit and use_8bit:
        raise ValueError("Both use_4bit and use_8bit cannot be True at the same time.")
    
    cls = bnb.nn.Linear4bit if use_4bit else (bnb.nn.Linear8bitLt if use_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
    """
    Calculate the number of warm-up steps for model training.

    Parameters:
    - epochs (int): Number of training epochs.
    - dataset_size (int): Total size of the training dataset.
    - batch_size (int): Batch size per device.
    - gradient_accumulation_steps (int): Number of gradient accumulation steps.
    - warmup_ratio (float): Ratio of warm-up steps to total training steps.

    Returns:
    - int: Number of warm-up steps.
    """
    steps_per_epoch = (dataset_size / batch_size)
    total_steps = epochs * steps_per_epoch / gradient_accumulation_steps
    total_steps_per_device = total_steps / torch.cuda.device_count()
    warmup_steps = int(total_steps_per_device * warmup_ratio)
    return warmup_steps


def merge_and_save_distcp(model, distcp_dir):
    """
    Merge and save distributed checkpoints.

    Parameters:
    - model (PreTrainedModel): Model to save.
    - distcp_dir (str): Directory containing distributed checkpoints.
    """
    state_dict = {
        "model": model.state_dict(),
    }
    DCP.load_state_dict(
        state_dict=state_dict,
        storage_reader=DCP.FileSystemReader(distcp_dir),
        no_dist=True,
    )
    model.load_state_dict(state_dict["model"])

    save_path = distcp_dir + '-merged.pth'
    torch.save({'model_state_dict': model.state_dict()}, save_path)


def train(args):
    """
    Train a QLoRA-adapted language model.

    Parameters:
    - args (Namespace): Command-line arguments.
    """
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.plm_name,
        args.use_4bit,
        args.bnb_4bit_quant_type,
        args.bnb_4bit_compute_dtype,
        args.use_double_quant,
        args.device_map
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # modules = find_all_linear_names(model, args.use_4bit, args.use_8bit)
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

    model = prepare_model_for_kbit_training(model, False)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(args.dataset_name)
    dataset = dataset.map(lambda example: preprocess_example(example, args.en_sign, args.ko_sign, args.max_length, tokenizer))

    metric = evaluate.load('sacrebleu')

    wandb.login(
        anonymous='never',
        key=WANDB_CLIENT_KEY,
        relogin=True,
        force=True
    )
    wandb.init(project=args.project_name, name=args.run_name)

    dataset_size = len(dataset['train'])
    warmup_steps = calculate_warmup_steps(
        epochs=args.num_epochs,
        dataset_size=dataset_size,
        batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        group_by_length=args.group_by_length,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
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
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        # eval_accumulation_steps=args.eval_accumulation_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        remove_unused_columns=args.remove_unused_columns
    )


    def preprocess_logits_for_metrics(logits, labels):
        """
        Preprocess logits for metric calculation.

        Parameters:
        - logits (Tensor): Model logits.
        - labels (Tensor): Ground truth labels.

        Returns:
        - tuple: Tuple containing preprocessed predictions and labels.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        label_ids = torch.where(labels != -100, labels, tokenizer.pad_token_id)
        return pred_ids, label_ids


    def compute_sacrebleu(p):
        preds, labels = p.predictions[0], p.label_ids

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"SacreBLEU": result["score"]}
        result = {k: round(v, 4) for k, v in result.items()}

        return result


    if args.use_fsdp:
        accelerator = Accelerator()
        trainer = accelerator.prepare(
            Seq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
                args=training_args,
                compute_metrics=compute_sacrebleu,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                data_collator=default_data_collator
            )
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            args=training_args,
            compute_metrics=compute_sacrebleu,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            data_collator=default_data_collator
        )
            
    trainer.train()
    
    if args.use_fsdp:
        trainer.model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(trainer.model)
        )
    else:
        trainer.model.save_pretrained(
            args.output_dir
        )


if __name__ == '__main__':
    yaml_path = './llama_config.yaml'
    args = parse_arguments_llama(yaml_path)
    train(args)
