# built-in
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# third-party
import torch
import wandb
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from peft import prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig

# custom
sys.path.append('../../../../')
from custom_utils.training_utils import set_seed
from custom_utils.general_secret import WANDB_CLIENT_KEY
from custom_utils.argument import parse_arguments_llama
from data_collator import CustomDataCollatorForCompletionOnlyLM


def load_model_and_tokenizer(
        plm_name,
        device_map,
        max_length,
        use_gradient_checkpointing,
        bnb_config,
        lora_path,
    ):
    model = AutoModelForCausalLM.from_pretrained(
        plm_name,
        quantization_config=bnb_config,
        torch_dtype=bnb_config.bnb_4bit_compute_dtype,
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
    model = PeftModel.from_pretrained(
        model, 
        lora_path, 
        torch_dtype=bnb_config.bnb_4bit_compute_dtype
    )
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
    model.print_trainable_parameters()

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
    lora_path = args.lora_path

    model, tokenizer = load_model_and_tokenizer(
        plm_name=args.plm_name, 
        # device_map=args.device_map, 
        device_map=torch.cuda.current_device(),
        max_length=args.max_length, 
        use_gradient_checkpointing=args.gradient_checkpointing,
        bnb_config=bnb_config, 
        lora_path=lora_path
    )
    dataset = load_dataset(args.dataset_name)

    metric = evaluate.load('sacrebleu')

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
        logging_steps = args.logging_steps
        eval_steps = warmup_steps
        save_steps = warmup_steps
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']

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
        per_device_eval_batch_size=args.per_device_batch_size,
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
    )

    data_collator=CustomDataCollatorForCompletionOnlyLM(
        # instruction_template='\n'.join([args.instruction, args.en_sign]),
        instruction_template=args.ko_sign,
        response_template=args.en_sign,
        tokenizer=tokenizer, 
        mlm=False
    )

    
    def formatting_func(example):
        output_texts = []
        for i in range(len(example['en'])):
            # text = f"{args.instruction}\n{args.en_sign} {example['en'][i]}\n{args.ko_sign} {example['ko'][i]}" # w/ instruction
            text = f"{args.ko_sign} {example['ko'][i]}\n{args.en_sign} {example['en'][i]}" # w/o instruction
            output_texts.append(text)
        return output_texts


    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels


    def compute_sacrebleu(p):
        IGNORE_INDEX = -100

        preds, labels = p.predictions[0], p.label_ids

        first_non_ignore_indices = np.argmax(labels != IGNORE_INDEX, axis=1)
        for i, idx in enumerate(first_non_ignore_indices):
            preds[i, :idx-1] = tokenizer.pad_token_id
        preds[preds == IGNORE_INDEX] = tokenizer.pad_token_id
        labels = np.where(labels != IGNORE_INDEX, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        for decoded_label, decoded_pred in zip(decoded_labels[:10], decoded_preds[:10]):
            print("[LABEL]")
            print(decoded_label[0])
            print("[PREDICTION]")
            print(decoded_pred)

        result = metric.compute(references=decoded_labels, predictions=decoded_preds)
        result = {"SacreBLEU": result["score"]}
        result = {k: round(v, 4) for k, v in result.items()}

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
        compute_metrics=compute_sacrebleu,
        max_seq_length=args.max_length,
        packing=False
    )
            
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    yaml_path = './llama_reverse_config.yaml'
    args = parse_arguments_llama(yaml_path)
    train(args)