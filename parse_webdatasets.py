import torch
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import webdataset as wds
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import argparse
import io

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def preprocess(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]).unsqueeze(0).to(device)

def identity(x):
    return x

def filter_dataset(item):
    if 'json' not in item:
        return False
    if 'jpg' not in item:
        return False
    return True


def main(clip_model_type: str, torch_device: str, webdataset_dir: str, output_filename: str, num_workers: int, batch_size: int):
    global device
    device = torch.device(torch_device)
    
    clip_model_name = clip_model_type.replace('/', '_')
    
    out_path = output_filename
    
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False, download_root='/mnt/theocoombes/huggingface-cache/clip')
    
    i = 0
    all_embeddings = []
    all_captions = []

    dataset = wds.WebDataset(webdataset_dir).select(filter_dataset).decode('pil').to_tuple('jpg', 'json').map_tuple(preprocess, identity)
    dataloader = wds.WebLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, prefetch_factor=4*batch_size)

    for images, jsn in tqdm(dataloader, desc="processing embeddings", unit="batch"):
        for i2 in range(len(images)):
            d = {}

            image = images[i2]

            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()

            captions = [jsn["caption"][i2]]

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
