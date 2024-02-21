## en2ko
# clean
echo "Translation mode: en2ko-clean"
nohup accelerate launch train_mt5.py \
    --translation_mode en2ko-clean \
    --output_dir ../models/base-fft-en2ko-clean \
    --run_name mt5-base-fft-en2ko \
    > train_mt5_base_fft_en2ko_clean.out 2>&1 &
wait
echo "Training done: en2ko-clean"

# separate
echo "Translation mode: en2ko-separate"
nohup accelerate launch train_mt5.py \
    --translation_mode en2ko-separate \
    --output_dir ../models/base-fft-en2ko-separate \
    --run_name mt5-base-fft-en2ko-separate \
    > train_mt5_base_fft_en2ko_separate.out 2>&1 &
wait
echo "Training done: en2ko-separate"

# separate-token
echo "Translation mode: en2ko-separate-token"
nohup accelerate launch train_mt5.py \
    --translation_mode en2ko-separate-token \
    --output_dir ../models/base-fft-en2ko-separate-token \
    --run_name mt5-base-fft-en2ko-separate-token \
    > train_mt5_base_fft_en2ko_separate_token.out 2>&1 &
wait
echo "Training done: en2ko-separate-token"

# first
echo "Translation mode: en2ko-first"
nohup accelerate launch train_mt5.py \
    --translation_mode en2ko-first \
    --output_dir ../models/base-fft-en2ko-first \
    --run_name mt5-base-fft-en2ko-first \
    > train_mt5_base_fft_en2ko_first.out 2>&1 &
wait
echo "Training done: en2ko-first"

# first-token
echo "Translation mode: en2ko-first-token"
nohup accelerate launch train_mt5.py \
    --translation_mode en2ko-first-token \
    --output_dir ../models/base-fft-en2ko-first-token \
    --run_name mt5-base-fft-en2ko-first-token \
    > train_mt5_base_fft_en2ko_first_token.out 2>&1 &
wait
echo "Training done: en2ko-first-token"

# second
echo "Translation mode: en2ko-second"
nohup accelerate launch train_mt5.py \
    --translation_mode en2ko-second \
    --output_dir ../models/base-fft-en2ko-second \
    --run_name mt5-base-fft-en2ko-second \
    > train_mt5_base_fft_en2ko_second.out 2>&1 &
wait
echo "Training done: en2ko-second"

# second-token
echo "Translation mode: en2ko-second-token"
nohup accelerate launch train_mt5.py \
    --translation_mode en2ko-second-token \
    --output_dir ../models/base-fft-en2ko-second-token \
    --run_name mt5-base-fft-en2ko-second-token \
    > train_mt5_base_fft_en2ko_second_token.out 2>&1 &
wait
echo "Training done: en2ko-second-token"


# ## ko2en
# # clean
# echo "Translation mode: ko2en-clean"
# nohup accelerate launch train_mt5.py \
#     --translation_mode ko2en-clean \
#     --output_dir ../models/base-fft-ko2en-clean \
#     --run_name mt5-base-fft-ko2en \
#     > train_mt5_base_fft_ko2en_clean.out 2>&1 &
# wait
# echo "Training done: ko2en-clean"

# # separate
# echo "Translation mode: ko2en-separate"
# nohup accelerate launch train_mt5.py \
#     --translation_mode ko2en-separate \
#     --output_dir ../models/base-fft-ko2en-separate \
#     --run_name mt5-base-fft-ko2en-separate \
#     > train_mt5_base_fft_ko2en_separate.out 2>&1 &
# wait
# echo "Training done: ko2en-separate"

# # separate-token
# echo "Translation mode: ko2en-separate-token"
# nohup accelerate launch train_mt5.py \
#     --translation_mode ko2en-separate-token \
#     --output_dir ../models/base-fft-ko2en-separate-token \
#     --run_name mt5-base-fft-ko2en-separate-token \
#     > train_mt5_base_fft_ko2en_separate_token.out 2>&1 &
# wait
# echo "Training done: ko2en-separate-token"

# # first
# echo "Translation mode: ko2en-first"
# nohup accelerate launch train_mt5.py \
#     --translation_mode ko2en-first \
#     --output_dir ../models/base-fft-ko2en-first \
#     --run_name mt5-base-fft-ko2en-first \
#     > train_mt5_base_fft_ko2en_first.out 2>&1 &
# wait
# echo "Training done: ko2en-first"

# # first-token
# echo "Translation mode: ko2en-first-token"
# nohup accelerate launch train_mt5.py \
#     --translation_mode ko2en-first-token \
#     --output_dir ../models/base-fft-ko2en-first-token \
#     --run_name mt5-base-fft-ko2en-first-token \
#     > train_mt5_base_fft_ko2en_first_token.out 2>&1 &
# wait
# echo "Training done: ko2en-first-token"

# # second
# echo "Translation mode: ko2en-second"
# nohup accelerate launch train_mt5.py \
#     --translation_mode ko2en-second \
#     --output_dir ../models/base-fft-ko2en-second \
#     --run_name mt5-base-fft-ko2en-second \
#     > train_mt5_base_fft_ko2en_second.out 2>&1 &
# wait
# echo "Training done: ko2en-second"

# # second-token
# echo "Translation mode: ko2en-second-token"
# nohup accelerate launch train_mt5.py \
#     --translation_mode ko2en-second-token \
#     --output_dir ../models/base-fft-ko2en-second-token \
#     --run_name mt5-base-fft-ko2en-second-token \
#     > train_mt5_base_fft_ko2en_second_token.out 2>&1 &
# wait
# echo "Training done: ko2en-second-token"


# ## mixed
# # clean
# echo "Translation mode: mixed-clean"
# nohup accelerate launch train_mt5.py \
#     --translation_mode mixed-clean \
#     --output_dir ../models/base-fft-mixed-clean \
#     --run_name mt5-base-fft-mixed-clean \
#     > train_mt5_base_fft_mixed_clean.out 2>&1 &
# wait
# echo "Training done: mixed-clean"

# # separate
# echo "Translation mode: mixed-separate"
# nohup accelerate launch train_mt5.py \
#     --translation_mode mixed-separate \
#     --output_dir ../models/base-fft-mixed-separate \
#     --run_name mt5-base-fft-mixed-separate \
#     > train_mt5_base_fft_mixed_separate.out 2>&1 &
# wait
# echo "Training done: mixed-separate"

# # separate-token
# echo "Translation mode: mixed-separate-token"
# nohup accelerate launch train_mt5.py \
#     --translation_mode mixed-separate-token \
#     --output_dir ../models/base-fft-mixed-separate-token \
#     --run_name mt5-base-fft-mixed-separate-token \
#     > train_mt5_base_fft_mixed_separate_token.out 2>&1 &
# wait
# echo "Training done: mixed-separate-token"

# # first
# echo "Translation mode: mixed-first"
# nohup accelerate launch train_mt5.py \
#     --translation_mode mixed-first \
#     --output_dir ../models/base-fft-mixed-first \
#     --run_name mt5-base-fft-mixed-first \
#     > train_mt5_base_fft_mixed_first.out 2>&1 &
# wait
# echo "Training done: mixed-first"

# # first-token
# echo "Translation mode: mixed-first-token"
# nohup accelerate launch train_mt5.py \
#     --translation_mode mixed-first-token \
#     --output_dir ../models/base-fft-mixed-first-token \
#     --run_name mt5-base-fft-mixed-first-token \
#     > train_mt5_base_fft_mixed_first_token.out 2>&1 &
# wait
# echo "Training done: mixed-first-token"

# # second
# echo "Translation mode: mixed-second"
# nohup accelerate launch train_mt5.py \
#     --translation_mode mixed-second \
#     --output_dir ../models/base-fft-mixed-second \
#     --run_name mt5-base-fft-mixed-second \
#     > train_mt5_base_fft_mixed_second.out 2>&1 &
# wait
# echo "Training done: mixed-second"

# # second-token
# echo "Translation mode: mixed-second-token"
# nohup accelerate launch train_mt5.py \
#     --translation_mode mixed-second-token \
#     --output_dir ../models/base-fft-mixed-second-token \
#     --run_name mt5-base-fft-mixed-second-token \
#     > train_mt5_base_fft_mixed_second_token.out 2>&1 &
# wait
# echo "Training done: mixed-second-token"

echo "All training done"
