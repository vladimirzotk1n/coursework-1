import pytest
import torch
from models import UNet
from coursework_1.config import *


@pytest.fixture()
def load_model():
    model = UNet(N_CHANNELS, NUM_CLASSES)
    model.to(DEVICE)
    weights = torch.load(UNET_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(weights)
    model.eval()
    return model


def test_model_loading(load_model):
    try:
        model = load_model

    except Exception as e:
        assert False, f"Не удалось загрузить модель {e}"


def test_model_parameters(load_model):
    model = load_model

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert 30_000_000 < num_parameters < 32_000_000


def test_model_shapes(load_model):
    model = load_model

    sample_data = torch.randn(BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)
    try:
        with torch.no_grad():
            pred = model(sample_data)

    except Exception as e:
        assert False, f"Не удалось сделать предсказание {e}"

    example_pred = torch.zeros((BATCH_SIZE, NUM_CLASSES, IMG_SIZE, IMG_SIZE))

    assert pred.shape == example_pred.shape