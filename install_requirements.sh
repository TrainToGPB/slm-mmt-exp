pip install -r requirements.txt

# conda install -c intel mkl_fft
# conda install -c intel mkl_random
# conda install -c intel mkl_service

pip install flash-attn --no-build-isolation
git clone https://github.com/kongds/MoRA.git
pip install -e ./MoRA/peft-mora

mkdir /data/.cache/hub/
mkdir /data/.cache/datasets/
