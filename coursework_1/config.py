import torch
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths:
CHECKPOINT_PATH = BASE_DIR / "models/checkpoint.pth"
TRAIN_DATASET_PATH = BASE_DIR / "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
UNET_WEIGHTS_PATH = BASE_DIR / "models/unet_weights.pth"
ARCHIVE_URL = "https://drive.google.com/uc?id=1QXmUdCwljYeJThjskDOsj4X4FF7My84v"
WEIGHTS_URL = "https://drive.google.com/uc?id=1EGnMJOfKqoTQsj_qQgVbcIG4OnGHsWbt"

# Dataset:
VOLUME_SLICES = 100
VOLUME_START_AT = 22
IMG_SIZE = 128
N_CHANNELS = 2
BATCH_SIZE = 16
NUM_CLASSES = 4

# Training:
PATIENCE = 5
NUM_EPOCHS = 35

