# Notebooks

Оригинальные исследовательские ноутбуки из проекта  
[Ecommerce_Customer_LTV_Prediction](https://github.com/Mamaeva-Bariyat/Ecommerce_Customer_LTV_Prediction/tree/main/notebooks) можно положить сюда.

Рекомендуемый порядок:
1. `1.LTV_Preprocessed_data.ipynb`
2. `2.EDA — Exploratory Data Analysis.ipynb`
3. `3.Геокластеризация, отзывы и RFM.ipynb`
4. `4.Feature_Engineeringg.ipynb`
5. `5.Modelingg_LTV.ipynb`

После выполнения ноутбуков 1–4 скопируйте `features_.csv` в `../data/processed/features_.csv` — тогда `python -m src.pipeline` сможет обучить модель.
