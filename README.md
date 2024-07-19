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
#### Model
- 현재 모델은 DGX H100 서버(114.110.129.131) 내 `/data/sehyeong/nmt/models/mmt_ft` 폴더에 존재

  - 가장 고성능 모델은 LLaMA3-KO에 MMT-Prime-300K를 학습시킨 모델
  - 어댑터: `/data/sehyeong/nmt/models/mmt_ft/ko-enjazh/ko`
  - 병합: `/data/sehyeong/nmt/models/mmt_ft/ko-enjazh/ko-merged`
  - 추후 61 서버로 모델 파일 이동 예정

#### Inference
- 추후 작성 예정

#### Demo (Streamlit)
- 추후 작성 예정
