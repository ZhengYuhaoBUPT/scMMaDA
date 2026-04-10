# coding=utf-8
# Copyright 2025 MMaDA Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import json
import math
import os
import random
import re
import pandas as pd
from functools import partial
from typing import List, Optional, Union

from PIL import Image

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

import webdataset as wds
import yaml
from braceexpand import braceexpand
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

person_token = ["a person", "someone", "somebody"]

def replace_person_token(t):
    "Used for CC12M - handles all case variations of <person> tag"
    t = re.sub(r"<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t, flags=re.IGNORECASE)
    
    person_pattern = re.compile(r"<person>", re.IGNORECASE)
    while person_pattern.search(t):
        match = person_pattern.search(t)
        t = t[:match.start()] + f" {random.choice(person_token)} " + t[match.end():]
    
    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None, src=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if "fname" not in filesample.keys():
            print(f"fname not in filesample.keys(): {filesample}")
            print(f"src: {src}")
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()

        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler) # [{fname,data,__url__}, ...]  __url__ 字段标识当前读取的文件来自哪个 tar 包
    samples = group_by_keys_nothrow(files, handler=handler, src=src)
    return samples


def image_transform(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    sample["images"] = image
    return sample

def image_transform_squash(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(image)
    sample["images"] = image
    return sample

def conditional_image_transform(sample, resolution=256):
    url = sample.get("__url__", "") 
    special_datasets = ['ai2d', 'clevr', 'docvqa', 'geo']
    use_squash = False
    for keyword in special_datasets:
        if keyword in url:
            use_squash = True
            break
    if use_squash:
        return image_transform_squash(sample, resolution)
    else:
        return image_transform(sample, resolution)


def remove_prefix(caption):
    caption = caption.replace('The image features ', '').replace('The image presents ', '').replace(
        "The image you've sent is, ", '').replace("In the center of the image, ", '').replace(
        "The image showcases ", '').replace("The image is ", '').replace(
        "The image captures ", '').replace("In the given image ", '').replace(
        "The image portrays ", '').replace("In the image, ", '').replace("In this image, we see ", '').replace(
        "The image depicts ", '').replace("This is ", '').replace("In this image, ", '').replace(
        "This image captures ", '')

    return caption

def filter_long_samples(sample):
    return sample.get('input_ids') is not None


class Text2ImageDataset:
    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str]],
            tokenizer: PreTrainedTokenizer,
            max_seq_length: int,
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            resolution: int = 256,
            shuffle_buffer_size: int = 1000,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            external_caption_path: Optional[str] = '',
            external_journeydb_caption_path: Optional[str] = '',
            external_laion12m_caption_path: Optional[str] = '',
            external_cc12m_caption_path: Optional[str] = '',
            external_text_to_image_2M_512_caption_path: Optional[str] = '',
            external_ai2d_caption_path: Optional[str] = '',
            external_clevr_caption_path: Optional[str] = '',
            external_docvqa_caption_path: Optional[str] = '',
            external_geo_caption_path: Optional[str] = '',
            is_captioning: bool = False,
            add_caption_prompt: bool = False,
            long_caption: bool = True,
            shuffle: bool = True,
    ):
        if f"{train_shards_path_or_url}.yaml" in os.listdir('./configs'):
            with open(f"./configs/{train_shards_path_or_url}.yaml") as f:
                train_shards_path_or_url = yaml.safe_load(f)
        self.long_caption = long_caption
        self.external_caption_path = external_caption_path
        self.external_journeydb_caption_path = external_journeydb_caption_path
        self.external_laion12m_caption_path = external_laion12m_caption_path
        self.external_cc12m_caption_path = external_cc12m_caption_path
        self.external_text_to_image_2M_512_caption_path = external_text_to_image_2M_512_caption_path
        self.is_captioning = is_captioning
        self.add_caption_prompt = add_caption_prompt
        if self.add_caption_prompt:
            with open("./training/questions.json") as f:
                self.caption_prompt = json.load(f)
                # self.caption_prompt = ['USER: \n' + prompt + ' ASSISTANT:' for prompt in self.caption_prompt]
                self.caption_prompt = ['<|start_header_id|>user<|end_header_id|>\n' + prompt + '<eot_id><|start_header_id|>assistant<|end_header_id|>\n' for prompt in self.caption_prompt]
        else:
            self.caption_prompt = None

        if external_journeydb_caption_path != '':
            with open(external_journeydb_caption_path) as file:
                self.journeydb_caption = json.load(file)
        else:
            self.journeydb_caption = None

        if external_ai2d_caption_path!= '':
            self.ai2d_caption = pd.read_csv(external_ai2d_caption_path)
        if external_clevr_caption_path!= '':
            self.clevr_caption = pd.read_csv(external_clevr_caption_path)
        if external_docvqa_caption_path!= '':
            self.docvqa_caption = pd.read_csv(external_docvqa_caption_path)
        if external_geo_caption_path!= '':
            self.geo_caption = pd.read_csv(external_geo_caption_path)

        def tokenize(text):
            if tokenizer is not None:
                text = replace_person_token(text)
                
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=2 * max_seq_length,
                    padding=False,
                    return_tensors="pt"
                )
                full_input_ids = encoding.input_ids[0]
                
                if len(full_input_ids) > max_seq_length:
                    return None
                else:
                    return text
            else:
                return text



        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        if external_caption_path != '':
            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.map(self.load_external_caption, handler=wds.ignore_and_continue),
                wds.rename(
                    images="jpg;png;jpeg;webp",
                    input_ids="text;txt;caption",
                    handler=wds.warn_and_continue,
                ),
                wds.map(partial(conditional_image_transform, resolution=resolution), handler=wds.warn_and_continue),
                wds.map(filter_keys(set(["images", "input_ids"]))),
                wds.map_dict(
                    input_ids=tokenize,
                    handler=wds.warn_and_continue,
                ),
                wds.select(filter_long_samples), 
            ]
        else:
            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.rename(
                    images="jpg;png;jpeg;webp",
                    input_ids="text;txt;caption",
                    handler=wds.warn_and_continue,
                ),
                wds.map(partial(conditional_image_transform, resolution=resolution), handler=wds.warn_and_continue),
                wds.map(filter_keys(set(["images", "input_ids"]))),
                wds.map_dict(
                    input_ids=tokenize,
                    handler=wds.warn_and_continue,
                ),
                wds.select(filter_long_samples),  
            ]

        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    def load_external_caption(self, sample):

        if 'SA1B' in sample['__key__'] or 'sa' in sample['__key__']:
            captionf = f"{self.external_caption_path}/{sample['__key__'].split('/')[-1]}.txt"
            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.readlines()[0].replace('\n', '')
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + captions
                else:
                    sample['txt'] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample['txt'] = captions.split('.')[0]
                else:
                    sample['txt'] = captions

                sample['txt'] = remove_prefix(sample['txt'])

            return sample

        elif 'laion' in sample['__url__']:
            url_part = sample['__url__'].split('/')[-1].split('.')[0] 
            key = sample['__key__'].split('/')[-1]  
            captionf = os.path.join(self.external_laion12m_caption_path, url_part, f"{key}.caption")

            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.read().strip()
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt  + captions
                else:
                    sample['txt'] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample['txt'] = captions.split('.')[0]
                else:
                    sample['txt'] = captions

                sample['txt'] = remove_prefix(sample['txt'])

            return sample

        elif 'cc12m' in sample['__url__']:
            url_part = sample['__url__'].split('/')[-1].split('.')[0]  
            key = sample['__key__'].split('/')[-1]  
            captionf = os.path.join(self.external_cc12m_caption_path, url_part, f"{key}.caption")

            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.read().strip()
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + captions
                else:
                    sample['txt'] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample['txt'] = captions.split('.')[0]
                else:
                    sample['txt'] = captions
                sample['txt'] = remove_prefix(sample['txt'])

            return sample

        elif "text-to-image-2M" in sample['__url__']:
            if "json" in sample and "prompt" in sample["json"]:
                captions = sample["json"]["prompt"]
            else:
                print(f"sample has no json or prompt: {sample}")
                captions = ""
    

            sample['txt'] = captions

            return sample

        elif 'ai2d' in sample['__url__']:
            key = sample['__key__'].split('/')[-1] 
            df_row = self.ai2d_caption[self.ai2d_caption['image'].astype(str) == key + '.png']
            if len(df_row) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample
            elif len(df_row) > 1:
                # print(f"Multiple captions available for key {sample['__key__']}")
                df_row = df_row.sample(1)
            question = df_row['question'].values[0]
            solution = df_row['solution'].values[0]
            caption = (
                '<|start_header_id|>user<|end_header_id|>\n'
                "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
                f"{solution}"
            )
            sample['txt'] = caption
            return sample

        elif 'clevr' in sample['__url__']:
            key = sample['__key__'].split('/')[-1]
            df_row = self.clevr_caption[self.clevr_caption['image'].astype(str) == key + ".jpg"]
            if len(df_row) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample
            elif len(df_row) > 1:
                # print(f"Multiple captions available for key {sample['__key__']}")
                df_row = df_row.sample(1)
            question = df_row['question'].values[0]
            solution = df_row['solution'].values[0]
            caption = (
                '<|start_header_id|>user<|end_header_id|>\n'
                "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
                f"{solution}"
            )
            sample['txt'] = caption
            return sample

        elif 'docvqa' in sample['__url__']:
            key = sample['__key__'].split('/')[-1]
            df_row = self.docvqa_caption[self.docvqa_caption['image'].astype(str) == key + ".png"]
            if len(df_row) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample
            elif len(df_row) > 1:
                # print(f"Multiple captions available for key {sample['__key__']}")
                df_row = df_row.sample(1)
            question = df_row['question'].values[0]
            solution = df_row['solution'].values[0]
            caption = (
                '<|start_header_id|>user<|end_header_id|>\n'
                "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
                f"{solution}"
            )
            sample['txt'] = caption
            return sample

        elif 'geo' in sample['__url__']:
            key = sample['__key__'].split('/')[-1]
            df_row = self.geo_caption[self.geo_caption['image'].astype(str) == key + ".jpg"]
            if len(df_row) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample
            elif len(df_row) > 1:
                # print(f"Multiple captions available for key {sample['__key__']}")
                df_row = df_row.sample(1)
            question = df_row['question'].values[0]
            solution = df_row['solution'].values[0]
            caption = (
                '<|start_header_id|>user<|end_header_id|>\n'
                "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
                f"{solution}"
            )
            sample['txt'] = caption
            return sample


        elif self.journeydb_caption is not None and sample['__key__'] in self.journeydb_caption:
            captions_list = self.journeydb_caption[sample['__key__']]
            if len(captions_list) == 0:
                print(f"No captions available for key {sample['__key__']}")
                return sample 
            sample['txt'] = random.sample(captions_list, 1)[0] 
            return sample

        else:
            print(f"none exist sample: {sample}")
            return sample 

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


if __name__ == '__main__':
    pass


class CellwTextDataset:
    """
    Dataset for CellwText single-cell multimodal data.
    Supports gene-to-text (g2t/mmug) training with gene expression data.
    """
    def __init__(
            self,
            lmdb_paths: Union[str, List[str]],
            gene_vocab_path: str,
            celltype_label_path: Optional[str] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            max_seq_length: int = 128,
            max_gene_tokens: int = 2000,
            num_expression_bins: int = 51,
            lmdb_vocab_path: Optional[str] = None,
            cell_metadata_path: Optional[str] = None,
            cell_feature_root: Optional[str] = None,
            caption_template: Optional[str] = None,
            batch_size: int = 4,
            num_workers: int = 4,
            shuffle: bool = True,
            pin_memory: bool = False,
    ):
        import lmdb

        if isinstance(lmdb_paths, list):
            self.lmdb_paths = lmdb_paths
        elif isinstance(lmdb_paths, str) and os.path.isdir(lmdb_paths):
            self.lmdb_paths = sorted([
                os.path.join(lmdb_paths, p)
                for p in os.listdir(lmdb_paths)
                if p.endswith('.db') and os.path.isdir(os.path.join(lmdb_paths, p))
            ])
        else:
            self.lmdb_paths = [lmdb_paths]
        self.max_seq_length = max_seq_length
        self.max_gene_tokens = max_gene_tokens
        self.num_expression_bins = num_expression_bins
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.caption_template = caption_template
        self.cell_feature_root = cell_feature_root
        self.h5ad_paths = {}
        self.h5ad_handles = {}
        self.h5ad_validated = set()

        if self.cell_feature_root is not None and os.path.isdir(self.cell_feature_root):
            self.h5ad_paths = {
                os.path.splitext(p)[0]: os.path.join(self.cell_feature_root, p)
                for p in os.listdir(self.cell_feature_root)
                if p.endswith('.h5ad')
            }

        # Load scgpt gene vocabulary
        with open(gene_vocab_path, 'r') as f:
            gene_vocab = json.load(f)
        self.gene_vocab = {gene: int(idx) for gene, idx in gene_vocab.items()}

        # Load LMDB vocabulary for ID mapping
        if lmdb_vocab_path is not None and os.path.exists(lmdb_vocab_path):
            with open(lmdb_vocab_path, 'r') as f:
                lmdb_vocab = json.load(f)
            self.lmdb_id2gene = {v: k for k, v in lmdb_vocab.items()}
            self.lmdb_id2scgpt_id = {
                lmdb_id: self.gene_vocab[gene_name]
                for lmdb_id, gene_name in self.lmdb_id2gene.items()
                if gene_name in self.gene_vocab
            }
        else:
            self.lmdb_id2scgpt_id = None

        # Optional celltype labels
        self.celltype_labels = None
        if celltype_label_path is not None and os.path.exists(celltype_label_path):
            with open(celltype_label_path, 'r') as f:
                self.celltype_labels = json.load(f)

        # Optional metadata table for captions: supports json/csv/tsv
        self.cell_metadata = None
        if cell_metadata_path is not None and os.path.exists(cell_metadata_path):
            if cell_metadata_path.endswith('.json'):
                with open(cell_metadata_path, 'r') as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    self.cell_metadata = {str(i): row for i, row in enumerate(loaded)}
                elif isinstance(loaded, dict):
                    self.cell_metadata = loaded
            elif cell_metadata_path.endswith('.csv') or cell_metadata_path.endswith('.tsv'):
                sep = '\t' if cell_metadata_path.endswith('.tsv') else ','
                df = pd.read_csv(cell_metadata_path, sep=sep)
                if 'idx' in df.columns:
                    self.cell_metadata = {str(row['idx']): row.to_dict() for _, row in df.iterrows()}
                elif 'index' in df.columns:
                    self.cell_metadata = {str(row['index']): row.to_dict() for _, row in df.iterrows()}
                else:
                    self.cell_metadata = {str(i): row.to_dict() for i, row in df.iterrows()}

        # Open LMDB environments and cache lengths
        self.envs = []
        self.env_lengths = []
        self.length = 0
        for lmdb_path in self.lmdb_paths:
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            self.envs.append(env)
            with env.begin() as txn:
                len_bytes = txn.get(b'__len__')
                env_len = int(len_bytes.decode('utf-8')) if len_bytes is not None else txn.stat()['entries']
            self.env_lengths.append(env_len)
            self.length += env_len

        print(f"CellwTextDataset initialized with {len(self.envs)} LMDB(s), total samples: {self.length}")

    def __len__(self):
        return self.length

    @staticmethod
    def _clean_optional_text(value):
        if value is None:
            return None
        value = str(value).strip()
        if value == '' or value.lower() in {'none', 'null', 'nan', 'unknown'}:
            return None
        return value

    def _get_celltype_label(self, idx):
        if self.celltype_labels is None:
            return None
        if isinstance(self.celltype_labels, dict):
            if str(idx) in self.celltype_labels:
                return self.celltype_labels[str(idx)]
            if idx in self.celltype_labels:
                return self.celltype_labels[idx]
        return None

    def _get_metadata_row(self, idx):
        if self.cell_metadata is None:
            return None
        if str(idx) in self.cell_metadata:
            return self.cell_metadata[str(idx)]
        if idx in self.cell_metadata:
            return self.cell_metadata[idx]
        return None

    @staticmethod
    def _first_non_empty(*values):
        for value in values:
            if value is None:
                continue
            value = str(value).strip()
            if value == '' or value.lower() in {'none', 'null', 'nan', 'unknown'}:
                continue
            return value
        return None

    @staticmethod
    def _truncate_text(text, max_chars=400):
        if text is None:
            return None
        text = str(text).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + '...'

    def _build_caption_from_record(self, data, metadata_row, celltype_label_raw):
        metadata_row = metadata_row if isinstance(metadata_row, dict) else {}

        celltype_name = self._first_non_empty(
            data.get('celltype_name'),
            data.get('celltype'),
            data.get('cell_type'),
            metadata_row.get('celltype_name'),
            metadata_row.get('celltype'),
            metadata_row.get('cell_type'),
            celltype_label_raw,
        )
        disease_name = self._first_non_empty(
            data.get('disease_name'),
            data.get('disease'),
            metadata_row.get('disease_name'),
            metadata_row.get('disease'),
        )
        tissue_name = self._first_non_empty(
            data.get('tissue_name'),
            data.get('tissue'),
            metadata_row.get('tissue_name'),
            metadata_row.get('tissue'),
        )
        sex_name = self._first_non_empty(data.get('sex_name'), metadata_row.get('sex_name'), metadata_row.get('sex'))
        stage_name = self._first_non_empty(data.get('stage_name'), metadata_row.get('stage_name'), metadata_row.get('stage'))

        celltype_def = self._truncate_text(self._first_non_empty(data.get('celltype_definition'), metadata_row.get('celltype_definition')))
        disease_def = self._truncate_text(self._first_non_empty(data.get('disease_definition'), metadata_row.get('disease_definition')))
        tissue_def = self._truncate_text(self._first_non_empty(data.get('tissue_definition'), metadata_row.get('tissue_definition')))

        parts = []
        base = "This cell"
        if celltype_name:
            base += f" is a {celltype_name}"
        if disease_name:
            base += f" under {disease_name} condition"
        if tissue_name:
            base += f" from {tissue_name}"
        if sex_name:
            base += f" of {sex_name}"
        if stage_name:
            base += f" at {stage_name} stage"
        if base != "This cell":
            parts.append(base + ".")

        if celltype_def:
            parts.append(f"Cell type definition: {celltype_def}")
        if disease_def:
            parts.append(f"Disease definition: {disease_def}")
        if tissue_def:
            parts.append(f"Tissue definition: {tissue_def}")

        if parts:
            return " ".join(parts)

        return self._build_caption(celltype_name, disease_name, tissue_name)

    def _build_caption(self, celltype, disease, tissue):
        celltype = self._clean_optional_text(celltype)
        disease = self._clean_optional_text(disease)
        tissue = self._clean_optional_text(tissue)

        if self.caption_template:
            disease_clause = f" under {disease} condition" if disease else ""
            tissue_clause = f" from {tissue}" if tissue else ""
            try:
                return self.caption_template.format(
                    celltype=celltype or "unknown cell",
                    disease=disease or "",
                    tissue=tissue or "",
                    disease_clause=disease_clause,
                    tissue_clause=tissue_clause,
                ).strip()
            except Exception:
                pass

        if not celltype and not disease and not tissue:
            return ""

        sentence = "This cell"
        if celltype:
            sentence += f" is a {celltype}"
        if disease:
            sentence += f" under {disease} condition"
        if tissue:
            sentence += f" from {tissue}"
        return sentence + "."

    def _resolve_env_and_local_idx(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset length {self.length}")
        running = 0
        for env_idx, env_len in enumerate(self.env_lengths):
            if idx < running + env_len:
                return env_idx, idx - running
            running += env_len
        raise IndexError(f"Index {idx} could not be resolved")

    def _get_h5ad_handle(self, env_idx):
        if not self.h5ad_paths:
            return None
        stem = os.path.splitext(os.path.basename(self.lmdb_paths[env_idx]))[0]
        if stem not in self.h5ad_paths:
            return None
        if stem not in self.h5ad_handles:
            import anndata as ad
            self.h5ad_handles[stem] = ad.read_h5ad(self.h5ad_paths[stem], backed='r')
        return self.h5ad_handles[stem]

    def _get_cell_feature(self, env_idx, local_idx):
        h5ad_handle = self._get_h5ad_handle(env_idx)
        if h5ad_handle is None:
            return None

        stem = os.path.splitext(os.path.basename(self.lmdb_paths[env_idx]))[0]
        if stem not in self.h5ad_validated:
            if local_idx >= h5ad_handle.n_obs:
                raise IndexError(
                    f"Local index {local_idx} out of range for H5AD shard {stem} with length {h5ad_handle.n_obs}"
                )
            lmdb_key = str(h5ad_handle.obs.iloc[local_idx]['lmdb_key']) if 'lmdb_key' in h5ad_handle.obs.columns else None
            expected_key = f"{local_idx:09d}"
            if lmdb_key is not None and lmdb_key != expected_key:
                raise ValueError(
                    f"H5AD/LMDB misalignment in shard {stem}: expected lmdb_key {expected_key}, got {lmdb_key}"
                )
            self.h5ad_validated.add(stem)

        import numpy as np
        import torch

        feature = np.asarray(h5ad_handle.X[local_idx]).reshape(-1)
        return torch.tensor(feature, dtype=torch.float32)

    def __getitem__(self, idx):
        import torch
        import msgpack

        # No zero-padding path: if a sample has fewer than max_gene_tokens valid mapped genes,
        # resample another index to keep fixed-length dense gene sequences.
        max_resample_attempts = 8
        cur_idx = int(idx)
        last_reason = None

        for _ in range(max_resample_attempts):
            env_idx, local_idx = self._resolve_env_and_local_idx(cur_idx)
            env = self.envs[env_idx]

            with env.begin() as txn:
                key10 = f'{local_idx:010d}'.encode('ascii')
                value = txn.get(key10)
                if value is None:
                    key9 = f'{local_idx:09d}'.encode('ascii')
                    value = txn.get(key9)
                if value is None:
                    last_reason = f"Index {cur_idx} not found in LMDB"
                    cur_idx = random.randint(0, self.length - 1)
                    continue
                try:
                    data = msgpack.unpackb(value, raw=False)
                except Exception:
                    try:
                        data = json.loads(value.decode('utf-8'))
                    except Exception as e:
                        last_reason = f"Failed to decode LMDB value at idx {cur_idx}: {e}"
                        cur_idx = random.randint(0, self.length - 1)
                        continue

            gene_ids_lmdb = list(data.get('gene_ids', []))
            log1p_x = list(data.get('log1p_x', []))

            if len(gene_ids_lmdb) == 0:
                last_reason = f"Empty gene_ids at idx {cur_idx}"
                cur_idx = random.randint(0, self.length - 1)
                continue

            if len(log1p_x) < len(gene_ids_lmdb):
                log1p_x = log1p_x + [0.0] * (len(gene_ids_lmdb) - len(log1p_x))

            pairs = list(zip(gene_ids_lmdb, log1p_x))

            # Map first, then take top-k by expression.
            # This avoids losing valid capacity when some of the top-k raw ids are unmapped.
            if self.lmdb_id2scgpt_id is not None:
                mapped_pairs_all = [
                    (self.lmdb_id2scgpt_id[lmdb_id], expr)
                    for lmdb_id, expr in pairs
                    if lmdb_id in self.lmdb_id2scgpt_id
                ]
            else:
                mapped_pairs_all = pairs

            if len(mapped_pairs_all) < self.max_gene_tokens:
                last_reason = (
                    f"Insufficient mapped genes at idx {cur_idx}: "
                    f"raw={len(gene_ids_lmdb)}, mapped_all={len(mapped_pairs_all)}, need={self.max_gene_tokens}"
                )
                cur_idx = random.randint(0, self.length - 1)
                continue

            topk_indices = sorted(
                range(len(mapped_pairs_all)),
                key=lambda i: mapped_pairs_all[i][1],
                reverse=True,
            )[:self.max_gene_tokens]
            mapped_pairs = [mapped_pairs_all[i] for i in topk_indices]
            gene_ids = [int(gid) for gid, _ in mapped_pairs]
            gene_expr = [float(expr) for _, expr in mapped_pairs]

            metadata_row = self._get_metadata_row(cur_idx)
            celltype_label_raw = self._get_celltype_label(cur_idx)
            caption_text = self._build_caption_from_record(data, metadata_row, celltype_label_raw)

            # Keep numeric tensor label for compatibility with existing code.
            if isinstance(celltype_label_raw, (int, float)):
                celltype_label_tensor = torch.tensor(int(celltype_label_raw), dtype=torch.long)
            else:
                celltype_label_tensor = torch.tensor(-1, dtype=torch.long)

            cell_feature = self._get_cell_feature(env_idx, local_idx)

            return {
                'gene_ids': torch.tensor(gene_ids, dtype=torch.long),
                'gene_expression': torch.tensor(gene_expr, dtype=torch.float32),
                'celltype_label': celltype_label_tensor,
                'texts': caption_text,
                'cell_features': cell_feature,
            }

        raise RuntimeError(f"Failed to fetch valid non-padded sample after {max_resample_attempts} attempts: {last_reason}")

    def collate_fn(self, batch):
        import torch

        gene_ids = torch.stack([item['gene_ids'] for item in batch])
        gene_expression = torch.stack([item['gene_expression'] for item in batch])
        celltype_label = torch.stack([item['celltype_label'] for item in batch])
        texts = [item.get('texts', '') for item in batch]
        cell_features = None
        if batch and batch[0].get('cell_features') is not None:
            cell_features = torch.stack([item['cell_features'] for item in batch])

        output = {
            'gene_ids': gene_ids,
            'gene_expression': gene_expression,
            'celltype_label': celltype_label,
            'texts': texts,
        }
        if cell_features is not None:
            output['cell_features'] = cell_features
        return output

    def get_dataloader(self):
        """
        Returns a DataLoader for this dataset.
        """
        from torch.utils.data import DataLoader

        sampler = None
        if self.shuffle:
            from torch.utils.data import RandomSampler
            sampler = RandomSampler(self)

        return DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def close(self):
        """Close all LMDB environments."""
        for env in self.envs:
            env.close()
        self.envs = []
