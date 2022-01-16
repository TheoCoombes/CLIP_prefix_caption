import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import webdataset as wds
import argparse

def imagetransform(b):
    return Image.fromarray(io.imread(filename))


def main(clip_model_type: str, device: str):
    device = torch.device(device)
    clip_model_name = clip_model_type.replace('/', '_')
    
    out_path = "alamy_embeddings.pkl"
    
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    
    images = list(Path("./images/").glob("*.jpg"))
    
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(images))):
        d = {}
        
        image_file = images[i]
        img_id = image_file.name.split(".")[0]
        
        image = imagetransform(image_file)
        image = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        
        caption_file = Path(str(image_file).replace(".jpg", ".txt"))
        with open(caption_file, "r") as f:
            captions = [caption.strip() for caption in f.readlines() if caption.strip() != ""]
        
        d["image_id"] = img_id
        d["clip_embedding"] = i
        d["captions"] = captions
        
        all_embeddings.append(prefix)
        all_captions.append(d)
        
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--device', default="cuda:0")
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.device))
