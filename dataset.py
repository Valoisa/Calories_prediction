import timm

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from PIL import Image

from transformers import AutoTokenizer

import pandas as pd

import numpy as np

import re


digit = re.compile('\d+')

def get_ingredients_dict(config):
    ingrs_df = pd.read_csv(config.VOCAB_DF_PATH)
    ingrs_dict = dict(zip(ingrs_df['id'].to_list(), ingrs_df['ingr'].to_list()))
    return ingrs_dict

def convert_to_recipe(text, ingrs_dict):
    ids_list = [int(digit.findall(ingr)[0]) for ingr in text.split(';')]
    ingrs_list = [ingrs_dict[idx] for idx in ids_list]
    return ', '.join(ingrs_list)


class MultimodalDataset(Dataset):
    def __init__(self, config, split_type='train'):
        super().__init__()
        ingrs_dict = get_ingredients_dict(config)

        self.df = pd.read_csv(
            config.TRAIN_TEST_DF_PATH,
            converters={
                'ingredients': lambda x: convert_to_recipe(x, ingrs_dict)
            }
            )
        split_types = self.df['split'].value_counts().index.to_list()
        if split_type not in split_type:
            raise ValueError(f'split_type must be one of these: {split_types}')
        self.df = self.df[self.df['split'] == split_type]

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        target = float(self.df.iloc[index]['total_calories'])
        mass = float(self.df.iloc[index]['total_mass'])
        text = self.df.iloc[index]['ingredients']

        dish_id = self.df.iloc[index]['dish_id']
        try:
            image = Image.open(f"data/images/{dish_id}/rgb.png").convert('RGB')
        except:
            image = torch.randint(0, 255, (*self.image_cfg.input_size[1:],
                                           self.image_cfg.input_size[0])).to(
                                               torch.float32)
        image = image=np.array(image)

        return {
            'target': target,
            'mass': mass,
            'text': text,            
            'image': image,
            'dish_id': dish_id
        }
    
def collate_fn(batch, tokenizer, transforms):
    targets = torch.tensor([item['target'] for item in batch])
    masses = torch.tensor([item['mass'] for item in batch])
    texts = [item['text'] for item in batch]
    images = torch.stack([transforms(image=item['image'])['image'] for item in batch])
    dish_ids = [item['dish_id'] for item in batch]

    input_ids = tokenizer(texts,
                          return_tensors='pt',
                          padding='max_length',
                          truncation=True)
    
    return {
        'target': targets,
        'mass': masses,
        'image': images,
        'input_ids': input_ids['input_ids'],
        'attention_mask': input_ids['attention_mask'],
        'dish_id': dish_ids
    }