import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

HF_CACHE_DIR = '/data/.cache/hub/'
MODEL_BASE_DIR = '/data/sehyeong/nmt/models'

SEED = 42
MAX_LENGTH = 1024

VLLM_VRAM_LIMIT = 0.95
VLLM_MAX_LORA_RANK = 64

MODEL_MAPPING = {
    # PLM
    'llama-2': 'meta-llama/Llama-2-7b-hf',
    'llama-2-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama-2-ko': 'beomi/open-llama-2-ko-7b',
    'llama-2-ko-chat': 'kfkas/Llama-2-ko-7b-Chat',
    'llama-3': 'meta-llama/Meta-Llama-3-8B',
    'llama-3-chat': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama-3-ko': 'beomi/Llama-3-Open-Ko-8B',
    'llama-3-ko-chat': 'beomi/Llama-3-Open-Ko-8B-Instruct-preview',

    # MMT-Seq2Seq
    'mbart': 'facebook/mbart-large-50-many-to-many-mmt',
    'madlad-7b': 'madlag/madlad-7b-mmt',

    # KO-EN
    'llama-2-ko-prime-base-en': 'traintogpb/llama-2-enko-translator-7b-qlora-bf16-upscaled',
    'llama-3-ko-prime-base-en': 'traintogpb/llama-3-enko-translator-8b-qlora-bf16-upscaled',

    # KO-EN-WORD
    'llama-3-ko-prime-base-en-word': os.path.join(MODEL_BASE_DIR, 'word/llama3-word-mix-all-bidir-merged-bf16'),

    # KO-JA
    'llama-3-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/baseline-merged'),
    'llama-3-ko-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/ko-merged'),
    'llama-3-ja-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/ja-merged'),
    'llama-3-koja-halfja-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/koja-halfja-merged'),
    'llama-3-koja-sigmoid-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/koja-sigmoid-merged'),
    'llama-3-koja-linear-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/koja-linear-merged'),

    # KO-ZH
    'llama-3-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/baseline-merged'),
    'llama-3-ko-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/ko-merged'),
    'llama-3-zh-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/zh-merged'),
    'llama-3-kozh-halfzh-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/kozh-halfzh-merged'),
    'llama-3-kozh-sigmoid-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/kozh-sigmoid-merged'),
    'llama-3-kozh-linear-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/kozh-linear-merged'),

    # MMT
    'llama-3-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/baseline-merged'),
    'llama-3-ko-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/ko-merged'),
    'llama-3-ja-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/ja-merged'),
    'llama-3-zh-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/zh-merged'),
    'llama-3-kojazh-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh-merged'),
    'llama-3-kojazh-halfall-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh-halfall-merged'),
    'llama-3-kojazh-equal-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh-equal-merged'),
    'llama-3-kojazh-halfjazh-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh-halfjazh-merged'),
    'llama-3-kojazh-quarterjazh-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh-quarterjazh-merged'),
}

PLM_MAPPING = {
    # Foundations
    'llama-2': 'meta-llama/Llama-2-7b-hf',
    'llama-2-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama-2-ko': 'beomi/open-llama-2-ko-7b',
    'llama-2-ko-chat': 'kfkas/Llama-2-ko-7b-Chat',
    'llama-3': 'meta-llama/Meta-Llama-3-8B',
    'llama-3-chat': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama-3-ko': 'beomi/Llama-3-Open-Ko-8B',
    'llama-3-ko-chat': 'beomi/Llama-3-Open-Ko-8B-Instruct-preview',

    # KO-JA Langvec
    'llama-3-koja-halfja': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-koja-langvec-scaled-half-ja'),
    'llama-3-koja-sigmoid': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-koja-langvec-scaled-sigmoid'),
    'llama-3-koja-linear': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-koja-langvec-scaled-linear'),
    
    # KO-ZH Langvec
    'llama-3-kozh-halfzh': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-kozh-langvec-scaled-half-zh'),
    'llama-3-kozh-sigmoid': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-kozh-langvec-scaled-sigmoid'),
    'llama-3-kozh-linear': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-kozh-langvec-scaled-linear'),

    # KO-JAZH Langvec
    'llama-3-kojazh': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-kojazh-langvec'),
    'llama-3-kojazh-halfall': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-kojazh-langvec-scaled-half-all'),
    'llama-3-kojazh-equal': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-kojazh-langvec-scaled-equal'),
    'llama-3-kojazh-halfjazh': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-kojazh-langvec-scaled-half-ja-zh'),
    'llama-3-kojazh-quarterjazh': os.path.join(MODEL_BASE_DIR, 'llama3-multilingual-kojazh-langvec-scaled-quarter-ja-zh'),
}

ADAPTER_MAPPING = {
    # KO-EN
    'llama-2-ko-prime-base-en': 'traintogpb/llama-2-enko-translator-7b-qlora-adapter',
    # 'llama-3-ko-prime-base-en': 'traintogpb/llama-3-enko-translator-8b-qlora-adapter',

    # KO-EN-WORD
    'llama-3-ko-prime-base-en-word': os.path.join(MODEL_BASE_DIR, 'word/llama3-word-mix-all-bidir'),

    # KO-JA
    # 'llama-3-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/baseline'),
    # 'llama-3-ko-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/ko'),
    'llama-3-ja-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/ja'),
    'llama-3-koja-halfja-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/koja-halfja'),
    'llama-3-koja-sigmoid-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/koja-sigmoid'),
    'llama-3-koja-linear-prime-base-ja': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-ja/koja-linear'),

    # KO-ZH
    'llama-3-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/baseline'),
    'llama-3-ko-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/ko'),
    'llama-3-zh-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/zh'),
    'llama-3-kozh-halfzh-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/kozh-halfzh'),
    'llama-3-kozh-sigmoid-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/kozh-sigmoid'),
    'llama-3-kozh-linear-prime-base-zh': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-zh/kozh-linear'),

    # MMT
    'llama-3-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/baseline'),
    'llama-3-ko-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/ko'),
    'llama-3-ja-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/ja'),
    'llama-3-zh-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/zh'),
    'llama-3-kojazh-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh'),
    'llama-3-kojazh-halfall-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh-halfall'),
    'llama-3-kojazh-equal-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh-equal'),
    'llama-3-kojazh-halfjazh-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh-halfjazh'),
    'llama-3-kojazh-quarterjazh-prime-base-mmt': os.path.join(MODEL_BASE_DIR, 'mmt_ft/ko-enjazh/kojazh-quarterjazh'),
}

DF_PATH_DICT = {
    'sample': os.path.join(SCRIPT_DIR, "../results/sample.csv"),
    'aihub': os.path.join(SCRIPT_DIR, "../results/test_tiny_uniform100_inferenced.csv"),
    'flores': os.path.join(SCRIPT_DIR, "../results/test_flores_inferenced.csv"),
    'prime-api': os.path.join(SCRIPT_DIR, "../results/prime/test_sparta_bidir_api_inferenced.csv"),
    'prime-llama2': os.path.join(SCRIPT_DIR, "../results/prime/test_sparta_bidir_llama2_inferenced.csv"),
    'prime-llama3': os.path.join(SCRIPT_DIR, "../results/prime/test_sparta_bidir_llama3_inferenced.csv"),
    'dpo': os.path.join(SCRIPT_DIR, "../results/koen_dpo_bidir_inferenced.csv"),
    'ko_words': os.path.join(SCRIPT_DIR, "../results/words/ko_word_test_1k.csv"),
    'en_words': os.path.join(SCRIPT_DIR, "../results/words/en_word_test_1k.csv"),
    'ja': os.path.join(SCRIPT_DIR, "../results/mmt/ja_test_bidir_inferenced.csv"),
    'zh': os.path.join(SCRIPT_DIR, "../results/mmt/zh_test_bidir_inferenced.csv"),
    'mmt': os.path.join(SCRIPT_DIR, "../results/mmt/mmt_test_bidir_inferenced.csv"),
    'mmt-m2m': os.path.join(SCRIPT_DIR, "../results/mmt/mmt_m2m_test_bidir_inferenced.csv"),
    'mmt-test': os.path.join(SCRIPT_DIR, "../results/mmt/mmt_test_bidir_inferenced.csv"),
}

DEEPL_LANG_CODE = {
    'en': {'src': 'EN', 'tgt': 'EN-US'},
    'ko': {'src': 'KO', 'tgt': 'KO'},
    'ja': {'src': 'JA', 'tgt': 'JA'},
    'zh': {'src': 'ZH', 'tgt': 'ZH'},
}
PAPAGO_LANG_CODE = {
    'en': 'en',
    'ko': 'ko',
    'ja': 'ja',
    'zh': 'zh-CN',
}
MBART_LANG_CODE = {
    'en': 'en_XX',
    'ko': 'ko_KR',
    'ja': 'ja_XX',
    'zh': 'zh_CN',
}
MADLAD_LANG_CODE = {
    'en': 'en',
    'ko': 'ko',
    'ja': 'ja',
    'zh': 'zh',
}
LLAMA_LANG_TABLE = {
    'en': 'English',
    'ko': '한국어',
    'ja': '日本語',
    'zh': '中文',
}
