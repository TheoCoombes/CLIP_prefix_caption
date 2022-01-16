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

def main(clip_model_type: str, device: str, webdataset_dir: str, output_filename: str):
    device = torch.device(device)
    clip_model_name = clip_model_type.replace('/', '_')
    
    out_path = output_filename
    
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False, download_root='/mnt/theocoombes/huggingface-cache/clip')
    
    tars = list(Path(webdataset_dir).glob("*.tar"))[:32]
    
    i = 0
    all_embeddings = []
    all_captions = []
    for tar in tqdm(tars, desc="generating embeddings from tars"):
        dataset = wds.WebDataset(str(tar.resolve()))
        
        for sample in dataset:
            d = {}

            image = imagetransform(sample["jpg"])
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()

            captions = [sample["caption"]]

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
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--webdataset-dir')
    parser.add_argument('--output-pkl-filename', default="embeddings.pkl")
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.device, args.webdataset_dir, args.output_pkl_filename))
