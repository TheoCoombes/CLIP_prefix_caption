""" A modified version of clip_inference.py from rom1504/clip-retrieval """

#!pip install clip-anytorch fire
import fire
from PIL import Image, UnidentifiedImageError
import json
import fsspec
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
import tqdm
import io

def preprocess_text_tokens(tokens: torch.Tensor, max_sequence_length: int, prefix_length: int):
    padding = max_sequence_length - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        tokens = tokens[:max_sequence_length]
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = 0
    mask = mask.float()
    mask = torch.cat((torch.ones(prefix_length), mask), dim=0)  # adding prefix mask
    return tokens, mask

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_image_dataset():
    """retrieve image dataset module without importing torch at the top level"""

    from torch.utils.data import Dataset  # pylint: disable=import-outside-toplevel

    class ImageDataset(Dataset):
        """ImageDataset is a pytorch Dataset exposing image and text tensors from a folder of image and text"""

        def __init__(self, preprocess, folder, tokenizer_model="gpt2", max_token_length=100, prefix_length=10):
            super().__init__()
            import clip  # pylint: disable=import-outside-toplevel
            from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

            path = Path(folder)

            text_files = [*path.glob("**/*.txt")]
            text_files = {text_file.stem: text_file for text_file in text_files}
            
            image_files = [
                *path.glob("**/*.png"),
                *path.glob("**/*.jpg"),
                *path.glob("**/*.jpeg"),
                *path.glob("**/*.bmp"),
            ]
            image_files = {image_file.stem: image_file for image_file in image_files}

            keys = None
            join = lambda new_set: new_set & keys if keys is not None else new_set
            
            keys = join(text_files.keys())
            keys = join(image_files.keys())

            self.keys = list(keys)
            
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            self.max_token_length = max_token_length
            self.prefix_length = prefix_length
            
            self.text_files = {k: v for k, v in text_files.items() if k in keys}

            self.image_files = {k: v for k, v in image_files.items() if k in keys}
            self.image_transform = preprocess

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, ind):
            key = self.keys[ind]
            output = {}

            try:
                image_file = self.image_files[key]
                image_tensor = self.image_transform(Image.open(image_file))
            except (UnidentifiedImageError, OSError):
                print(f"Failed to load image {image_file}. Skipping.")
                return None  # return None to be filtered in the batch collate_fn

            output["image_filename"] = str(image_file)
            output["image_tensor"] = image_tensor

            text_file = self.text_files[key]
            caption = text_file.read_text()
            
            text_tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
            text_tokens, mask = preprocess_text_tokens(text_tokens, self.max_token_length, self.prefix_length)
            
            output["text_tokens"] = np.concatenate((text_tokens.numpy(), mask.numpy()))
            output["text"] = caption

            return output

    return ImageDataset


def create_webdataset(
    urls,
    image_transform,
    image_key="jpg",
    caption_key="txt",
    caption_in_metadata=False,
    cache_path=None,
    tokenizer_model="gpt2",
    max_token_length=100,
    prefix_length=10
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    import clip  # pylint: disable=import-outside-toplevel
    import webdataset as wds  # pylint: disable=import-outside-toplevel
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def filter_dataset(item):
        if caption_key not in item and not caption_in_metadata:
            return False
        if image_key not in item:
            return False
        if caption_in_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        output = {}

        image_data = item[image_key]
        image = Image.open(io.BytesIO(image_data))
        image_tensor = image_transform(image)
        output["image_filename"] = item["__key__"]
        output["image_tensor"] = image_tensor

        if not caption_in_metadata:
            text = item[caption_key]
            caption = text.decode("utf-8")
            
            text_tokens = torch.tensor(tokenizer.encode(caption), dtype=torch.int64)
            text_tokens, mask = preprocess_text_tokens(text_tokens, max_token_length, prefix_length)
            
            output["text_tokens"] = np.concatenate((text_tokens.numpy(), mask.numpy()))
            output["text"] = caption
        else:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            caption = json.loads(metadata)[caption_key]
            
            text_tokens = torch.tensor(tokenizer.encode(caption), dtype=torch.int64)
            text_tokens, mask = preprocess_text_tokens(text_tokens, max_token_length, prefix_length)
            
            output["text_tokens"] = np.concatenate((text_tokens.numpy(), mask.numpy()))
            output["text"] = caption
        
        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


class OutputSink:
    """This output sink can save image, text embeddings as npy and metadata as parquet"""

    def __init__(self, output_folder, write_batch_size):
        self.fs, output_folder = fsspec.core.url_to_fs(output_folder)
        self.output_folder = output_folder
        self.img_emb_folder = output_folder + "/img_embeddings"
        self.text_token_folder = output_folder + "/text_tokens"
        self.metadata_folder = output_folder + "/metadata"

        if not self.fs.exists(self.output_folder):
            self.fs.mkdir(self.output_folder)
            batch_init_num = -1
        else:
            existing_top_level_files = self.fs.walk(self.metadata_folder).__next__()[2]
            if len(existing_top_level_files) == 0:
                batch_init_num = -1
            else:
                batch_init_num = max(
                    [int(x.split("/")[-1].split(".")[0].split("_")[1]) for x in existing_top_level_files]
                )
        if not self.fs.exists(self.img_emb_folder):
            self.fs.mkdir(self.img_emb_folder)

        if not self.fs.exists(self.text_token_folder):
            self.fs.mkdir(self.text_token_folder)
        
        if not self.fs.exists(self.metadata_folder):
            self.fs.mkdir(self.metadata_folder)

        self.write_batch_size = write_batch_size
        self.batch_count = 0
        self.batch_num = batch_init_num
        self.__init_batch()

    def __init_batch(self):
        self.image_embeddings = []
        self.text_tokens = []
        self.image_names = []
        self.captions = []
        self.metadata = []
        self.batch_count = 0
        self.batch_num += 1

    def add(self, image_embs, text_tokens, image_filenames, captions):
        """
        add to buffers the image embeddings, text embeddings, and meta
        """
        self.batch_count += image_embs.shape[0]
        self.image_embeddings.append(image_embs)
        self.image_names.extend(image_filenames)
        self.captions.extend(captions)
        self.text_tokens.extend(text_tokens)

        if self.batch_count > self.write_batch_size:
            self.flush()

    def __write_batch(self):
        """
        write a batch of embeddings and meta to npy and parquet
        """
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pandas as pd  # pylint: disable=import-outside-toplevel

        data_lists = []
        data_columns = []

        img_emb_mat = np.concatenate(self.image_embeddings)
        output_path_img = self.img_emb_folder + "/img_emb_" + str(self.batch_num)

        with self.fs.open(output_path_img + ".npy", "wb") as f:
            npb = BytesIO()
            np.save(npb, img_emb_mat)
            f.write(npb.getbuffer())

        data_lists.append(self.image_names)
        data_columns.append("image_path")

        text_token_mat = np.array(self.text_tokens)
        output_path_text = self.text_token_folder + "/text_emb_" + str(self.batch_num)

        with self.fs.open(output_path_text + ".npy", "wb") as f:
            npb = BytesIO()
            np.save(npb, text_token_mat)
            f.write(npb.getbuffer())

        data_lists.append(self.captions)
        data_columns.append("caption")

        df = pd.DataFrame(data=list(zip(*data_lists)), columns=data_columns)

        output_path_metadata = self.metadata_folder + "/metadata_" + str(self.batch_num) + ".parquet"
        with self.fs.open(output_path_metadata, "wb") as f:
            df.to_parquet(f)

    def flush(self):
        if self.batch_count == 0:
            return
        self.__write_batch()
        self.__init_batch()


def clip_inference(
    input_dataset,
    output_folder,
    input_format="files",
    cache_path=None,
    batch_size=256,
    num_prepro_workers=8,
    write_batch_size=10 ** 6,
    subset_size=None,
    wds_image_key="jpg",
    wds_caption_key="txt",
    wds_caption_in_metadata=False,
    clip_model="ViT-B/32",
    text_tokenizer_model="gpt2",
    max_token_length=100,
    prefix_length=10,
    device="cuda:0"
):
    """clip inference goes from a image text dataset to clip embeddings"""

    import clip  # pylint: disable=import-outside-toplevel
    from torch.utils.data import DataLoader  # pylint: disable=import-outside-toplevel
    from torch.utils.data.dataloader import default_collate  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    model, preprocess = clip.load(clip_model, device=device, jit=False)
    model_img = model.encode_image

    if input_format == "files":
        dataset = get_image_dataset()(preprocess, input_dataset, tokenizer_model=text_tokenizer_model, max_token_length=max_token_length, prefix_length=prefix_length)
    elif input_format == "webdataset":
        dataset = create_webdataset(
            input_dataset,
            preprocess,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            caption_in_metadata=wds_caption_in_metadata,
            cache_path=cache_path,
            tokenizer_model=text_tokenizer_model,
            max_token_length=max_token_length,
            prefix_length=prefix_length
        )
    else:
        raise Exception(f"No such input format {input_format}")

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn if input_format == "files" else None,
    )
    output_sink = OutputSink(output_folder, write_batch_size)

    c = 0
    bar = tqdm.tqdm()
    for item in data:
        with torch.no_grad():
            image_features = model_img(item["image_tensor"].to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_embs = image_features.cpu().numpy()
            image_filename = item["image_filename"]

            text_tokens = item["text_tokens"]
            text = item["text"]

            output_sink.add(image_embs, text_tokens, image_filename, text)

        bar.update(batch_size)
        c += batch_size
        if subset_size is not None and c >= subset_size:
            break
    output_sink.flush()


if __name__ == "__main__":
    fire.Fire(clip_inference)
