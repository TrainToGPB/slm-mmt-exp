## enko_translation

### Purpose
1. LLaMA 2 기반 영-한 번역 모델 실험 및 QLoRA 활용
2. 파파고, 구글, 딥엘 등 상용 번역 API와의 성능 비교

### Usage
#### 0. Basic settings
  __a. mkl-fft 설치__
  ```bash
  conda install -c intel mkl_fft
  ```

  __b. requirements 설치__
  ```
  pip install -r requirements.txt
  ```

#### 1. `app.py` 이용 (recommended)
```bash
streamlit run ./app.py --server.fileWatcherType=none --server.port=30001
```

#### 2. `model_inferences.py` 이용
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

__`--inference_type`__ (default: `sentence`)
- `sentence`: 입력 문장 번역
- `dataset`: csv 데이터셋 번역

__`--dataset`__ (default: `sample`)
- `--inference_type=dataset`인 경우 사용 가능
- `sample`: `sample_texts_for_inferences.csv`
- `aihub`: `inference/results/test_tiny_uniform100_inferenced.csv` (사용 불가, 업로드 안 돼있음)
- `flores`: `inference/results/test_flores_inferenced.csv` (사용 불가, 업로드 안 돼있음)
- 따로 추가한 파일의 경로 입력 가능

__`--sentence`__ (default: `"NMIXX is a South Korean girl group that made a comeback on January 15, 2024 with their new song 'DASH'."`)
- `--inference_type=sentence`인 경우 사용 가능
- 번역하고자 하는 영어 문장 자유롭게 입력

