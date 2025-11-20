import pytest
import torch
from coursework_1.dataset import MRIDataset, get_dataloaders, split_data
from coursework_1.config import N_CHANNELS, IMG_SIZE, BATCH_SIZE, NUM_CLASSES


def test_split_data():
    train_data, val_data, test_data = split_data()
    assert len(train_data) > 0, "Длина train_data <= 0"
    assert len(val_data) > 0, "Длина val_data <= 0"
    assert len(test_data) > 0, "Длина test_data <= 0"
    
    assert isinstance(train_data, list), "train_data - не экземпляр list"
    assert isinstance(val_data, list), "val_data - не экземпляр list"
    assert isinstance(test_data, list), "test_data - не экземпляр list"
    
    assert all(isinstance(x, str) for x in train_data), "Элементы train_data - не строки"
    assert all(isinstance(x, str) for x in val_data), "Элементы val_data - не строки"
    assert all(isinstance(x, str) for x in test_data), "Элементы test_data - не строки"
    
 
def test_datasets():
    train_data, val_data, test_data = split_data()

    train_dataset = MRIDataset(train_data)
    val_dataset = MRIDataset(val_data)
    test_dataset = MRIDataset(test_data)

        
    assert len(train_dataset) > 0, "Длина train_dataset <= 0"
    assert len(val_dataset) > 0, "Длина val_dataset <= 0"
    assert len(test_dataset) > 0, "Длина test_dataset <= 0"

    example_data = torch.zeros((N_CHANNELS, IMG_SIZE, IMG_SIZE))
    assert train_dataset[0][0].shape == example_data.shape and \
            val_dataset[0][0].shape == example_data.shape and \
            test_dataset[0][0].shape == example_data.shape, "Размеры dataset неправильные"
    
    assert not torch.isnan(train_dataset[0][0]).any() and not torch.isinf(train_dataset[0][0]).any(), "train_dataset содержит inf или None"
    assert not torch.isnan(val_dataset[0][0]).any() and not torch.isinf(val_dataset[0][0]).any(), "val_dataset содержит inf или None"
    assert not torch.isnan(test_dataset[0][0]).any() and not torch.isinf(test_dataset[0][0]).any(), "test_dataset содержит inf или None"


def test_dataloaders():
    train_data, val_data, test_data = split_data()

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_data, val_data, test_data)
    img_example_data = torch.zeros((BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE))
    mask_example_data = torch.zeros((BATCH_SIZE, NUM_CLASSES, IMG_SIZE, IMG_SIZE))

    try:
        for train_imgs, train_masks in train_dataloader:
            break

        for val_imgs, val_masks in val_dataloader:
            break

        for test_imgs, test_masks in test_dataloader:
            break

    except Exception as e:
        assert False, f"Не удалось загрузить dataloader {e}"
    
    assert train_imgs.shape == img_example_data.shape and \
            train_masks.shape == mask_example_data.shape, "Размеры train_dataloader неправильные"

    assert val_imgs.shape == img_example_data.shape and \
            val_masks.shape == mask_example_data.shape, "Размеры val_dataloader неправильные"
    
    assert test_imgs.shape == img_example_data.shape and \
            test_masks.shape == mask_example_data.shape, "Размеры test_dataloader неправильные"
    
   

