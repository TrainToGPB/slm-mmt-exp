import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_weight(model_name, log_dir):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    writer = SummaryWriter(log_dir=log_dir)
    for name, param in tqdm(model.named_parameters(), total=len(list(model.named_parameters())), desc=f"Checking weight for {model_name}"):
        writer.add_histogram(name, param, 0)
    writer.close()
    print(f"Check the weight of the model in tensorboard: tensorboard --logdir={log_dir.split('/')[-1]}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--langs', type=str, nargs='+', required=True, help='Lang vector language')
    args = parser.parse_args()

    for lang in args.langs:
        model_name = f'../models/llama3-lang-vector-{lang}'
        log_dir = f'./tensorboard_logs/{model_name.split("/")[-1]}'  # 모델 이름을 기반으로 로그 디렉토리 설정
        check_weight(model_name, log_dir)
