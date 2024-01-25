# built-in
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# third-party
import torch
import wandb
import evaluate
import numpy as np
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig

# custom
sys.path.append('../../../../')
from custom_utils.training_utils import set_seed
from custom_utils.general_secret import WANDB_CLIENT_KEY
from custom_utils.argument import parse_arguments_llama


def load_model_and_tokenizer(
        plm_name,
        max_length,
        bnb_config,
        lora_config,
    ):

    model = AutoModelForCausalLM.from_pretrained(
        plm_name,
        quantization_config=bnb_config,
        torch_dtype=bnb_config.bnb_4bit_compute_dtype,
        device_map=torch.cuda.current_device()
    )
    model.config.use_cache = False
    model.config.pretrining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(plm_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = max_length

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model = prepare_model_for_kbit_training(model, False)
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def calculate_warmup_steps(epochs, dataset_size, batch_size, gradient_accumulation_steps, warmup_ratio):
    steps_per_epoch = (dataset_size / batch_size)
    total_steps = epochs * steps_per_epoch / gradient_accumulation_steps
    total_steps_per_device = total_steps / torch.cuda.device_count()
    warmup_steps = int(total_steps_per_device * warmup_ratio)
    return warmup_steps


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def train(args):
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

    model, tokenizer = load_model_and_tokenizer(args.plm_name, args.max_length, bnb_config, lora_config)
    dataset = load_dataset(args.dataset_name)

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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
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
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
    )

    data_collator=DataCollatorForCompletionOnlyLM(
        instruction_template=args.en_sign,
        response_template=args.ko_sign,
        tokenizer=tokenizer, 
        mlm=False
    )


    def formatting_func(example):
        output_texts = []
        for i in range(len(example['en'])):
            text = f"{args.en_sign} {example['en'][i]}\n {args.ko_sign} {example['ko'][i]}"
            output_texts.append(text)
        return output_texts


    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels


    def compute_sacrebleu(p):
        preds, labels = p.predictions[0], p.label_ids
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        print(preds.shape)
        print(preds)
        print(labels.shape)
        print(labels)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        print(decoded_labels[0])
        print(decoded_preds[0])

        result = metric.compute(references=decoded_labels, predictions=decoded_preds)
        result = {"SacreBLEU": result["score"]}
        result = {k: round(v, 4) for k, v in result.items()}

        return result


    if args.use_fsdp:
        accelerator = Accelerator()
        trainer = accelerator.prepare(
            SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
                args=training_args,
                data_collator=data_collator,
                formatting_func=formatting_func,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=compute_sacrebleu,
                max_seq_length=args.max_length,
                packing=False
            )
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            args=training_args,
            data_collator=data_collator,
            formatting_func=formatting_func,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_sacrebleu,
            max_seq_length=args.max_length,
            packing=False
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
