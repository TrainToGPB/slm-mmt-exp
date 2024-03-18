## enko_translation

### Purpose
1. LLaMA 2 기반 영-한 번역 모델 실험 및 QLoRA 활용
2. 파파고, 구글, 딥엘 등 상용 번역 API와의 성능 비교

### Usage
#### Install Dependencies
```bash
bash ./install_requirements.sh
```

### 1. Inference
#### 1-1. `app.py` 이용 (recommended)
```bash
streamlit run ./app.py --server.fileWatcherType=none --server.port=30001
```

#### 1-2. `model_inferences.py` 이용
```bash
python ./inference/codes/model_inferences.py
```
__Arguments__

__`--model_type`__ (default: `llama-bf16`)
- `mbart`: [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
- `nllb`: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- `madlad`: [google/madlad400-3b-mt](https://huggingface.co/google/madlad400-3b-mt)
- `llama`: [LLaMA-QLoRA finetuned](https://huggingface.co/traintogpb/llama-2-en2ko-translator-7b-qlora-adapter)
- `llama-bf16`: [LLaMA-QLoRA finetuned + BF16 upscaled & merged](https://huggingface.co/traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled)
- `llama-bf16-vllm`: [LLaMA-QLoRA finetuned + BF16 upscaled & merged](https://huggingface.co/traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled) + vLLM inference
- `llama-sparta`: [LLaMA-QLoRA finetuned](https://huggingface.co/traintogpb/llama-2-enko-translator-7b-qlora-adapter) with [traintogpb/aihub-flores-koen-integrated-sparta-30k](https://huggingface.co/datasets/traintogpb/aihub-flores-koen-integrated-sparta-30k)

__`--inference_type`__ (default: `sentence`)
- `sentence`: 입력 문장 번역
- `dataset`: csv 데이터셋 번역

__`--dataset`__ (default: `sample`)
- `--inference_type=dataset`인 경우 사용 가능
- `sample`: `sample_texts_for_inferences.csv`
- `aihub`: `inference/results/test_tiny_uniform100_inferenced.csv` (사용 불가, 업로드 안 돼있음)
- `flores`: `inference/results/test_flores_inferenced.csv` (사용 불가, 업로드 안 돼있음)
- `sparta`: `inference/results/test_sparta_bidir_inferenced.csv` (사용 불가, 업로드 안 돼있음)
- 따로 추가한 파일의 경로 입력 가능

__`--sentence`__ (default: `"NMIXX is a South Korean girl group that made a comeback on January 15, 2024 with their new song 'DASH'."`)
- `--inference_type=sentence`인 경우 사용 가능
- 번역하고자 하는 영어 문장 자유롭게 입력

### 2. Training
#### 2-1. LLaMA QLoRA SFT
```bash
accelerate launch \
    --main_process_port 50001 \
    --config_file ./training/llama_qlora/configs/deepspeed_train_config_bf16.yaml \
    --num_processes 4 \
    ./training/llama_qlora/codes/train_llama_sft.py \
    --wandb_key YOUR_WANDB_KEY \
    --dataloader_num_workers 4 \
    --output_dir ./training/llama_qlora/models/baseline-sparta \
    --per_device_batch_size 16 \
    --num_epochs 2 \
    --warmup_ratio 0.10 \
    --lr_scheduler_type constant_with_warmup \
    --optim paged_adamw_32bit \
    --gradient_accumulation_steps 4 \
    --logging_steps 25 \
    --load_best_model_at_end true \
    --metric_for_best_model sacrebleu_en2ko \
    --project_name sft_translation \
    --run_name llama-qlora-sparta \
    --just_test false
```
- `num_process`: GPU 개수 변경
- `wandb_key`: 본인의 WandB API 키 사용
  - `project_name`: WandB project 이름
  - `run_name`: WandB project 내 run 이름
- `just_test` (true인 경우):
  - train 데이터셋 1,000개, eval 데이터셋 10개만 사용
  - WandB의 `project_name`을 test로 고정
  - `logging_steps`, `eval_steps`, `save_steps`를 1로 고정
  - `output_dir`를 `./training/llama_qlora/models/test`로 고정
- 그 외 학습 configuration은 `./training/llama_qlora/configs/`의 `llama_config.yaml`에서 조정할 수 있음