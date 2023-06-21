import argparse
from huggingface_hub import HfApi


parser = argparse.ArgumentParser()
parser.add_argument('--local_folder', type=str, default='final_train_A100_v1')
parser.add_argument('--repo_id', type=str, default='colbert-team/colbert-vqa')
args = parser.parse_args()

if __name__ == '__main__':
    hf_api = HfApi()    
    hf_api.upload_folder(
        folder_path=args.local_folder,
        repo_id=args.repo_id)