# 📈 Ecommerce Customer LTV Prediction — MLOps

Production-ready MLOps-версия проекта прогнозирования **Lifetime Value (LTV)** клиентов на данных Olist (Brazilian E-Commerce).

Оригинальный research-проект: [Mamaeva-Bariyat/Ecommerce_Customer_LTV_Prediction](https://github.com/Mamaeva-Bariyat/Ecommerce_Customer_LTV_Prediction)

---

## 🎯 Бизнес-задача

- 95–99% клиентов делают только одну покупку.
- Нужно предсказать вероятность возврата (`will_return`) и оценить будущую ценность.
- **Формула**: `LTV ≈ P(return) × средний повторный чек`.
- Сегментация на 10 децилей → персонализированные маркетинговые стратегии.

**Модель**: ансамбль **XGBoost + LightGBM + CatBoost** (StratifiedKFold, балансировка классов).

---

## 🏗 Архитектура

```text
Ecommerce_LTV_MLOps/
├── config/config.yaml          # гиперпараметры и пути
├── src/
│   ├── data/                   # download, preprocess, features
│   ├── models/                 # train, predict, evaluate
│   ├── utils/                  # io, metrics
│   └── pipeline.py             # end-to-end обучение
├── backend/                    # FastAPI (serving)
├── frontend/                   # Streamlit dashboard
├── models/                     # ltv_ensemble.joblib (после обучения)
├── reports/                    # OOF-предсказания, метрики, lift
├── data/
│   ├── raw/                    # сырые CSV Olist (после download)
│   └── processed/              # features_.csv (после preprocess)
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## 🚀 Быстрый старт (macOS)

> На Mac используй **`python3`**, не `python`.  

### 1. Перейти в папку проекта

```bash
cd "/Users/<твой_user>/Downloads/Ecommerce_LTV_MLOps 2"
# или куда ты распаковал архив
pwd
ls
```

### 2. Создать и активировать виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

В начале строки должно появиться `(.venv)`.

### 3. Установить зависимости

```bash
pip install -U pip
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 4. Скачать данные Olist (Kaggle)

Нужен файл `~/.kaggle/kaggle.json` (API-токен с [kaggle.com/settings](https://www.kaggle.com/settings)).

```bash
python3 -m src.data.download
```

Ожидаемый результат:
```text
Downloading olistbr/brazilian-ecommerce → data/raw
Download & unzip completed.
```

### 5. Обучить модель (preprocess + train)

```bash
python3 -m src.pipeline
```

Пайплайн:
1. Собирает признаки из сырых CSV → `data/processed/features_.csv`
2. Обучает ансамбль XGBoost + LightGBM + CatBoost
3. Сохраняет артефакты:
   - `models/ltv_ensemble.joblib`
   - `reports/oof_predictions.csv`
   - `reports/cv_metrics.csv`
   - `reports/lift_table.csv`

    
### 6. Запустить API (терминал 1)

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

- Swagger: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health → должно быть `"model_loaded": true`

### 7. Запустить Streamlit (терминал 2)

```bash
source .venv/bin/activate
streamlit run frontend/main.py
```

Откроется: **http://localhost:8501**

---

## 🖥 Streamlit: как загрузить файл и получить LTV

### Вкладка «🔮 Скоринг»

1. В сайдбаре API URL = `http://localhost:8000`
2. Нажми **Browse files** и выбери CSV
3. Нажми **«Рассчитать LTV»**

### Какой CSV нужен

Файл должен содержать те же признаки, что при обучении  
(колонки из `data/processed/features_.csv`, кроме `will_return` и `future_ltv` — их можно оставить, модель их не использует).

**Где лежит полный файл:**
```text
data/processed/features_.csv
```

**Как быстро сделать тестовый sample:**
```bash
head -n 21 data/processed/features_.csv > sample_for_score.csv
```

Загрузи `sample_for_score.csv` на дашборде.

### Что появится после скоринга

- Метрики: средний `pred_LTV`, средняя `P(return)`, число клиентов
- Гистограмма распределения предсказанного LTV
- Распределение по децилям (1–10)
- Таблица с `pred_proba`, `pred_ltv`, `ltv_decile`
- Кнопка **«Скачать результаты (CSV)»**

### Другие вкладки

| Вкладка | Содержание |
|---------|------------|
| **📊 Сегменты** | Рекомендации по децилям LTV |
| **ℹ️ Модель** | OOF-метрики, avg_return_amount, список признаков |
| **📖 Рекомендации** | Бизнес-таблица по группам 1–3 / 4–7 / 8–10 |

---

---

## 📊 Рекомендации по сегментам

| Группа | Средний LTV | Рекомендация |
|--------|-------------|--------------|
| **1–3** | низкий | Минимальные усилия: общие рассылки 1 раз в квартал |
| **4–7** | средний | Персонализация + скидка 15% |
| **8–10** | высокий | Максимум внимания: индивидуальные предложения, VIP |

---

---

## 🔧 Конфигурация

Все настройки — в `config/config.yaml`:
- пути к данным и моделям
- гиперпараметры XGB / LightGBM / CatBoost
- `avg_return_amount`
- количество децилей
- seed

---

## 🐳 Docker (опционально)

```bash
docker-compose up --build
```

- API: http://localhost:8000  
- Frontend: http://localhost:8501  

---

## 🧪 Тесты

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## 📦 Стек

- **ML**: pandas, scikit-learn, xgboost, lightgbm, catboost, joblib  
- **Serving**: FastAPI, uvicorn  
- **UI**: Streamlit, plotly  
- **Infra**: Docker, Makefile  

---

---

*Автор MLOps-версии на основе проекта Mamaeva Bariyat*  
*Стек: Python · FastAPI · Streamlit · XGBoost · LightGBM · CatBoost · Docker*
