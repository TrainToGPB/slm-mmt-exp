## MMT-SPARTA-LARGE

### Purpose
- LLaMA를 기반으로 한 SPARTA-large 다국어 번역 모델 개발

### Model Configuration
- PLM: [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

  - 위 모델에서 한/일/중 언어를 continually pretrain한 다음 모델들도 활용
  - KO: [`beomi/Llama-3-Open-Ko-8B`](https://huggingface.co/beomi/Llama-3-Open-Ko-8B)
  - JA: [`rinna/llama-3-youko-8b`](https://huggingface.co/rinna/llama-3-youko-8b)
  - ZH: [`hfl/llama-3-chinese-8b`](https://huggingface.co/hfl/llama-3-chinese-8b)

- Dataset: AI Hub의 한-영, 한-일, 한-중 병렬 번역 말뭉치 데이터 병합 활용

  - 총 병합 데이터

    - 한-영: 총 10M
    - 한-일: 총 4.3M
    - 한-중: 총 5.9M

  - 고품질 데이터: [xCOMET](https://huggingface.co/Unbabel/XCOMET-XL) 점수에 기반한 고품질 데이터 추출

    - 한-영, 한-일, 한-중 각 100K 씩 추출해, MMT-Prime-300K 데이터셋 구성

  - Inference: Paged KV caching 기반 [vLLM](https://github.com/vllm-project/vllm) 툴 활용

    - 단일 추론 시: 약 115 tokens/sec (Transformers 약 35 tokens/sec)
    - 배치 추론 시: 약 4000 tokens/sec (128 배치 기준)

### Usage
#### Requirements
```bash
bash install_requirements.sh
```

#### Model
- 현재 모델은 DGX H100 서버(114.110.129.131) 내 `/data/sehyeong/nmt/models/mmt_ft` 폴더에 존재

  - 가장 고성능 모델은 LLaMA3-KO에 MMT-Prime-300K를 학습시킨 모델
  - 어댑터: `/data/sehyeong/nmt/models/mmt_ft/ko-enjazh/ko`
  - 병합: `/data/sehyeong/nmt/models/mmt_ft/ko-enjazh/ko-merged`
  - 추후 61 서버로 모델 파일 이동 예정

#### Train
- 코드를 동작시키기 전에, `./training/llama/codes/train_llama_sft.py` 파일 54번 째 줄에 본인의 WandB client key를 입력해야 함

```bash
accelerate launch --main_process_port=30001 --config_file ./training/llama/configs/deepspeed_train_config_bf16.yaml ./training/llama/codes/train_llama_sft.py \
  --plm_name beomi/Llama-3-Open-Ko-8B \
  --output_dir ./training/llama/models/test \
  --per_device_batch_size 4 \
  --eos_token_id 128001 \
  --pad_token_id 128002 \
  --metric_for_best_model sacrebleu_en2ko \
  --train_dataset_name YOUR_DATASET_PATH \
  --eval_dataset_name YOUT_DATASET_PATH \
  --project_name test \
  --run_name test \
  --just_test
```

- 그 외 argument는 `./training/llama/configs/llama_config.yaml` 파일에서 확인 가능
  - 위 YAML 파일을 수정하는 것으로도 학습 시 argument 변경 사항 적용 가능
- `just_test`의 경우, 데이터셋의 극히 일부만 사용하여 간단히 테스트 하는 용도일 때만 사용합니다.

#### Inference
- 공통 argument

  - `model_name`: `./inference/codes/translation_info.py`에 prefix된 간소화된 모델 이름
  - `lora_name`: `./inference/codes/translation_info.py`에 prefix된 간소화된 LoRA 어댑터 이름 (없는 경우 None)
  - `lora_nickname`: 모델에 LoRA 어댑터를 붙일 때 명명할 이름 (아무 이름이나 상관 없음; 없는 경우 None)
  - `model_type`: `api`, `hf`, `hf-qlora`, `vllm` 중 모델과 호환되는 방식 선택
  - `prompt_type`: 번역 프롬프트가 필요한 경우, `llama`, `madlad` 중 선택
    - 프롬프트 필요 없이 문장을 그대로 입력하는 경우 `None` 입력
    - `llama`: `"Translate this from {SRC_LANG} to {TGT_LANG}.\n### {SRC_LANG}: {SRC_TEXT}\n### {TGT_LANG}:"`
    - `madlad`: `"<2{TGT_LANG}> {SRC_TEXT}"`
  - `data_type`: 문장 번역의 경우 `sentence`, CSV 데이터셋 번역의 경우 `dataset` 입력

- Sentence translation
  ```bash
  python ./inference/codes/translation_inference.py \
    --model_name YOUR_MODEL_NAME \
    --lora_name YOUR_LORA_NAME \
    --lora_nickname YOUR_LORA_NICKNAME \
    --model_type vllm \
    --prompt_type llama \
    --data_type sentence \
    --sentence "Hello, world!" \
    --src_lang en \
    --tgt_lang ko
  ```
  - `sentence`: 번역할 원본 문장
  - `src_lang`: 번역 출발 언어
  - `tgt_lang`: 번역 도착 언어

- Dataset (CSV) translation
  ```bash
  python translation_inference.py \
    --model_name YOUR_MODEL_TYPE \
    --lora_name YOUR_LORA_NAME \
    --lora_nickname YOUR_LORA_NICKNAME \
    --model_type vllm \
    --prompt_type llama \
    --data_type dataset \
    --dataset_name mmt-sample \
    --tgt_col src \
    --lang_col direction \
    --src_lang None \
    --tgt_lang None \
    --trans_col trans \
    --time_col trans_time \
    --batch_size 128 \
    --print_result True
  ```
  - `dataset_name`: `./inference/codes/translation_info.py`에 prefix된 간소화된 번역 대상 CSV 데이터셋 이름, 또는 실제 CSV 데이터셋 경로
  - `tgt_col`: 데이터셋 내 번역 대상 column 이름
  - `lang_col`: 번역 방향(e.g., `en-ko`)을 지정한 column 이름 (존재하는 경우)
    - `lang_col`을 지정한 경우 `src_lang`과 `tgt_lang`은 None으로 설정
  - `src_lang`: 번역 출발 언어
    - `src_lang`을 지정한 경우 `lang_col`은 None으로 설정, `tgt_lang` 필수로 설정
  - `tgt_lang`: 번역 도착 언어
    - `tgt_lang`을 지정한 경우 `lang_col`은 None으로 설정, `src_lang` 필수로 설정
  - `trans_col`: 번역문을 기록할 column 이름
  - `time_col`: 번역 시간을 기록할 column 이름
  - `batch_size`: 추론 배치 사이즈
  - `print_result`: `True`인 경우 모든 데이터셋 내 텍스트 번역 결과 출력

#### Demo (Streamlit)
- 추후 작성 예정
