conda install -c intel mkl_fft
conda install -c intel mkl_random

FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn

pip install -r requirements.txt
