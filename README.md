Сегментация опухоли мозга на наборе данных BraTS

## 📁 Структура проекта
```
.
├── Makefile
├── README.md
├── data
│   └──       <- После загрузки тут будут данные.
├── models
│   ├── unet.py
│   └── unet_weights.pth      <- Веса модели после загрузки.
|
├── notebooks
│   ├── coursework.ipynb      <- Главный ноутбук   
│   └── coursework_dirty.ipynb      <- Ноутбук со всем кодом, можно запустить его в Google Colab отдельно
├── pyproject.toml
├── reports
│   └── figures
│       └──       <- Графики для отчета?
└── coursework_1
    ├── __init__.py      
    ├── config.py       
    ├── dataset.py      
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py
    │   ├── train.py
    │   ├── early_stopping.py
    │   ├── losses.py
    │   └── metrics.py
    └── plots.py

```
--------

Как скачать:
```
git clone https://github.com/vladimirzotk1n/coursework-1
cd coursework-1
uv sync
```
