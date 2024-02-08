nohup python train_llama_sft_reverse.py \
    --output_dir=../models/continuous-reverse-overlap \
    --dataset_name=traintogpb/aihub-koen-translation-integrated-tiny-100k \
    --run_name=llama-qlora-continuous-reverse-overlap \
    > train_llama_reverse_overlap.out 2>&1 &

wait

nohup python train_llama_sft_reverse.py \
    --output_dir=../models/continuous-reverse-new \
    --dataset_name=traintogpb/aihub-koen-translation-integrated-tiny-100k-2 \
    --run_name=llama-qlora-continuous-reverse-new \
    > train_llama_reverse_new.out 2>&1 &
