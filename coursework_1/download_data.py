from pathlib import Path
import gdown
import zipfile
from coursework_1.config import BASE_DIR, TRAIN_DATASET_PATH, UNET_WEIGHTS_PATH, ARCHIVE_URL, WEIGHTS_URL


def load_data():
    archive_path = BASE_DIR / "data/archive.zip"
    weights_path = UNET_WEIGHTS_PATH

    if not archive_path.exists():
        print("Скачиваем архив с данными...")
        gdown.download(ARCHIVE_URL, str(archive_path), quiet=False)

    if not weights_path.exists():
        print("Скачиваем веса модели...")
        gdown.download(WEIGHTS_URL, str(weights_path), quiet=False)

    if not TRAIN_DATASET_PATH.exists():
        print("Распаковываем архив...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR / "data")
        print("Готово!")

