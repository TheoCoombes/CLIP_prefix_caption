import torch
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import webdataset as wds
from pathlib import Path
import argparse
import io

def imagetransform(b):
    return Image.open(io.BytesIO(b))

def filter_dataset(item):
      if 'json' not in item:
          return False
      if 'jpg' not in item:
          return False
      return True


def main(clip_model_type: str, device: str, webdataset_dir: str, output_filename: str, num_workers: int, batch_size: int):
    device = torch.device(device)
    clip_model_name = clip_model_type.replace('/', '_')
    
    out_path = output_filename
    
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False, download_root='/mnt/theocoombes/huggingface-cache/clip')
    
    i = 0
    all_embeddings = []
    all_captions = []

    dataset = wds.WebDataset(webdataset_dir).select(filter_dataset).decode('rgb').to_tuple('jpg', 'json')
    dataloader = wds.WebLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, prefetch_factor=4*batch_size)

    for image, jsn in tqdm(dataloader, desc="processing embeddings"):
        d = {}

        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        captions = [jsn["caption"]]

        d["clip_embedding"] = i
        d["captions"] = captions

        all_embeddings.append(prefix)
        all_captions.append(d)

        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

        i += 1

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--worker-num', type=int, default=8)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--webdataset-dir')
    parser.add_argument('--output-pkl-filename', default="embeddings.pkl")
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.device, args.webdataset_dir, args.output_pkl_filename, args.worker_num, args.batch_size))
