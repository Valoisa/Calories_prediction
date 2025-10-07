import re

import timm

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchmetrics

from transformers import AutoModel, AutoTokenizer

from transformers import AutoTokenizer

import albumentations as A

import numpy as np

from functools import partial

from dataset import MultimodalDataset, collate_fn

        
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


def get_train_transforms(cnn_config):
    # Небольшие аугментации для изображений, а также
    # обязательная нормализация, изменение размера
    # и конвертация в тензор
    return A.Compose([
                A.SmallestMaxSize(max(cnn_config.input_size[1], cnn_config.input_size[2]), p=1),
                A.RandomCrop(height=cnn_config.input_size[1], width=cnn_config.input_size[2], p=1),
                A.Affine(scale=(0.75, 1.25),
                        rotate=(-30, 30), fill=255,
                        translate_percent=(-0.1, 0.1),
                        shear=(-10, 10),
                        p=0.3),
                A.CoarseDropout(num_holes_range=(2, 4), 
                                hole_height_range=(0.03*cnn_config.input_size[1], 0.07*cnn_config.input_size[1]),
                                hole_width_range=(0.03*cnn_config.input_size[2], 0.07*cnn_config.input_size[2]),
                                fill=255,
                                p=0.4),
                A.HueSaturationValue(hue_shift_limit=(-30, 30),
                                    sat_shift_limit=(-30, 30),
                                    val_shift_limit=(-20, 20), 
                                    p=0.4),
                A.Normalize(mean=cnn_config.mean, std=cnn_config.std),    
                A.ToTensorV2()
            ])

def get_test_transforms(img_config):
    # Обязательная нормализация, изменение размера
    # и конвертация в тензор
    return A.Compose([
                A.SmallestMaxSize(max(img_config.input_size[1], img_config.input_size[2]), p=1),
                A.RandomCrop(height=img_config.input_size[1], width=img_config.input_size[2], p=1),
                A.Normalize(mean=img_config.mean, std=img_config.std),    
                A.ToTensorV2()
            ])

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

        fused_emb = text_emb * image_emb
        
        output = self.regressor(fused_emb)
        output = torch.mul(output, mass.view(-1, 1))
        return output


def train(config, device):
    torch.manual_seed(config.SEED)
    
    # Инициализация модели
    model = MultimodalModel(config)
    set_requires_grad(model.text_model, config.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, config.IMAGE_MODEL_UNFREEZE)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # Оптимайзер 
    optimizer = torch.optim.AdamW([
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR},
        {'params': model.regressor.parameters(), 'lr': config.REGRESSOR_LR}
    ])

    # Для задачи регрессии выберем среднеквадратичную ошибку
    criterion = nn.MSELoss(reduction='mean')

    # Для трейна — трансформации с аугментациями, для валидации — без
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    transforms = get_train_transforms(cfg)
    val_transforms = get_test_transforms(cfg)

    # Загрузка данных
    train_val_dataset = MultimodalDataset(config, split_type='train')
    full_length = len(train_val_dataset)
    train_length = int(full_length*0.8)
    generator = torch.Generator().manual_seed(config.SEED)
    train_dataset, val_dataset = random_split(train_val_dataset, 
                                              [
                                                  train_length, 
                                                  full_length - train_length
                                              ],
                                              generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, transforms=transforms)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, transforms=val_transforms)
    )
    
    # Цикл обучения
    train_loss = 0.0
    best_mae = np.inf
    # Для валидации выберем MAE
    mae_metrics = torchmetrics.MeanAbsoluteError().to(device)
    for epoch in range(config.EPOCHS):
        model.train()
        mae_metrics.reset()
        for batch in train_loader:
            targets = batch['target'].to(device)
            masses = batch['mass'].to(device)
            images = batch['image'].to(device)
            text = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(**{'input_ids': text, 'attention_mask': attention_mask, 'image': images, 'mass': masses})
            loss = criterion(outputs.view(-1), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch {epoch + 1} train loss: {train_loss:.4f}')
        train_loss = 0.
        val_mae = validate(model, val_loader, device, mae_metrics)
        print(f'Val MAE: {val_mae:.4f}')

        if val_mae < best_mae:
            torch.save(model.state_dict(), config.SAVE_PATH)
            best_mae = val_mae

    # Результаты обучения на тестовой выборке
    model.load_state_dict(torch.load(config.SAVE_PATH, weights_only=True))
    model.to(device)
    test_dataset = MultimodalDataset(config, split_type='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, transforms=val_transforms)
    )
    test_mae = validate(model, test_loader, device, mae_metrics)
    print(f'Test MAE after {config.EPOCHS} epochs of train: {test_mae:.4f}')    


def validate(model, val_loader, device, mae_metrics):
    model.eval()
    mae_metrics.reset()
    val_mae = 0.
    with torch.no_grad():        
        for batch in val_loader:
            targets = batch['target'].to(device)
            masses = batch['mass'].to(device)
            images = batch['image'].to(device)
            text = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            preds = model(**{'input_ids': text, 'attention_mask': attention_mask, 'image': images, 'mass': masses})

            _ = mae_metrics(preds=preds.view(-1), target=targets)

        val_mae = mae_metrics.compute().cpu().numpy()
    return val_mae


def test_model_inferece(config, device):
    # Загрузим самую удачную модель
    model = MultimodalModel(config)
    model.load_state_dict(torch.load(config.SAVE_PATH))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    test_transforms = get_test_transforms(cfg)

    test_dataset = MultimodalDataset(config, split_type='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer, transforms=test_transforms)
        )
    mae_metrics = torchmetrics.MeanAbsoluteError().to(device)

    mae_metrics.reset()
    test_mae = 0.
    # Сохраним идентификаторы блюд и ошибки в предсказаниях, 
    # чтобы получить данные с самой большой ошибкой
    errors = []
    dish_ids = []
    with torch.no_grad():        
        for batch in test_loader:
            targets = batch['target'].to(device)
            masses = batch['mass'].to(device)
            images = batch['image'].to(device)
            text = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dish_ids.extend(batch['dish_id'])

            preds = model(**{'input_ids': text, 'attention_mask': attention_mask, 'image': images, 'mass': masses})
            diffs = (preds.view(-1) - targets).cpu().numpy()
            errors.extend(diffs)

            _ = mae_metrics(preds=preds.view(-1), target=targets)

        test_mae = mae_metrics.compute().cpu().numpy()
        abs_errors = np.abs(errors)
        indices = abs_errors.argsort()[::-1]
        hardest_ids = np.array(dish_ids)[indices[:5]]
        largest_errors = np.array(abs_errors)[indices[:5]]
    return test_mae, hardest_ids, largest_errors