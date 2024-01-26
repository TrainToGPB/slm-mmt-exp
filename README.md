# enko_translation

영-한 번역 모델 실험 및 QLoRA 활용
파파고, 구글, 딥엘 등 상용 번역 API와의 성능 비교

## 추론 코드 테스트
1. `git checkout test/function-test`
2. `conda install -c intel mkl_fft`
3. `pip install -r requirements.txt`
4. (API 추론 시) `enko_translation/inference/codes/api_secret.py` 에 API 정보 기재
5. __[API 추론 코드]__
    - `en_text` dafault: `NMIXX is a South Korean girl group that made a comeback on January 15, 2024 with their new song 'DASH'`.
    - `api_type` default: `google`
        - `google`, `deepl`, `papago` 가능
    ```bash
    python /enko_translation/inference/codes/api_inferences.py --en_text='ENGLISH_TEXT' --api_type='API_TYPE'
    ```
6. __[모델 추론 코드]__
    - `en_text` default: API 추론 코드와 동일
    - `model_type` default: `mbart`
        - `mbart`, `nllb-600m`, `madlad`
    ```bash
    python /enko_translation/inference/codes/model_inferences.py --en_text='ENGLISH_TEXT' --model_type'MODEL_TYPE'
    ```
