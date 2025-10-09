import re

import timm

import torch
import torch.nn as nn

from transformers import AutoModel

FUSION_TYPES = ['mul', 'sum', 'concat']

def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for param, _ in module.named_parameters():
            param.requires_grad = False
        return

    pattern = re.compile(unfreeze_pattern)

    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0 
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        if config.FUSION not in FUSION_TYPES:
            raise ValueError(f'Fusion type must be one of these: {FUSION_TYPES}.') 
        self.fusion = config.FUSION

        if self.fusion == 'concat':
            pass
            self.regressor = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM), 
                nn.LayerNorm(config.HIDDEN_DIM),       
                nn.ReLU(),                           
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2), 
                nn.LayerNorm(config.HIDDEN_DIM // 2),       
                nn.ReLU(),                           
                nn.Dropout(config.DROPOUT),                    
                nn.Linear(config.HIDDEN_DIM // 2, 1)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2), 
                nn.LayerNorm(config.HIDDEN_DIM // 2),       
                nn.ReLU(),                           
                nn.Dropout(config.DROPOUT),                    
                nn.Linear(config.HIDDEN_DIM // 2, 1)
            )


    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused_emb = None 
        if self.fusion == 'mul':
            fused_emb = text_emb * image_emb
        elif self.fusion == 'sum':
            fused_emb = text_emb + image_emb
        else:
            fused_emb = torch.cat((text_emb, image_emb), dim=-1)
        
        output = self.regressor(fused_emb)
        output = torch.mul(output, mass.view(-1, 1))
        return output