"""
Streamlit dashboard for Customer LTV Prediction.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Customer LTV Dashboard",
    page_icon="📈",
    layout="wide",
)

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

st.title("📈 Customer Lifetime Value (LTV) Prediction")
st.markdown(
    """
Прогноз вероятности повторной покупки и LTV клиентов e-commerce (Olist).  
Модель: ансамбль **XGBoost + LightGBM + CatBoost**.
"""
)

# ---------- Sidebar ----------
st.sidebar.header("Настройки")
show_raw = st.sidebar.checkbox("Показать сырые предсказания", value=False)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Скоринг", "📊 Сегменты", "ℹ️ Модель", "📖 Рекомендации"]
)

with tab1:
    st.subheader("Загрузка клиентов для скоринга")
    uploaded = st.file_uploader("CSV с признаками (тот же набор, что при обучении)", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"Загружено строк: **{len(df):,}**")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Рассчитать LTV", type="primary"):
            with st.spinner("Отправка на API..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                    if resp.status_code != 200:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                    else:
                        result = pd.DataFrame(resp.json())
                        st.success("Готово!")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Средний pred_LTV", f"{result['pred_ltv'].mean():.1f} R$")
                        col2.metric("Средняя P(return)", f"{result['pred_proba'].mean():.3f}")
                        col3.metric("Клиентов", f"{len(result):,}")

                        # Distribution
                        fig = px.histogram(
                            result,
                            x="pred_ltv",
                            nbins=40,
                            title="Распределение предсказанного LTV",
                            labels={"pred_ltv": "Predicted LTV (R$)"},
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Decile counts
                        decile_counts = result["ltv_decile"].value_counts().sort_index()
                        fig2 = px.bar(
                            x=decile_counts.index.astype(str),
                            y=decile_counts.values,
                            title="Распределение по децилям LTV",
                            labels={"x": "LTV Decile", "y": "Count"},
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                        if show_raw:
                            st.dataframe(result, use_container_width=True)

                        csv = result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Скачать результаты (CSV)",
                            csv,
                            "ltv_predictions.csv",
                            "text/csv",
                        )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Не удалось подключиться к API. "
                        "Запустите: `uvicorn backend.main:app --reload --port 8000`"
                    )
                except Exception as e:
                    st.error(str(e))
    else:
        st.info("Загрузите CSV с признаками клиентов, чтобы получить предсказания LTV.")

with tab2:
    st.subheader("Сегменты и рекомендации")
    try:
        resp = requests.get(f"{API_URL}/segments", timeout=10)
        if resp.status_code == 200:
            segs = pd.DataFrame(resp.json())
            st.dataframe(segs, use_container_width=True, hide_index=True)
        else:
            st.warning("API недоступен — показываю статические рекомендации.")
            raise Exception
    except Exception:
        static = pd.DataFrame(
            [
                {"decile": "1–3", "mean_LTV": "~0–60 R$", "share": "~30%", "recommendation": "Минимальные усилия: общие рассылки 1 раз в квартал"},
                {"decile": "4–7", "mean_LTV": "~60–100 R$", "share": "~40%", "recommendation": "Персонализация + скидка 15%"},
                {"decile": "8–10", "mean_LTV": "~100–140+ R$", "share": "~30%", "recommendation": "Максимум внимания: индивидуальные предложения, VIP"},
            ]
        )
        st.dataframe(static, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Информация о модели")
    try:
        resp = requests.get(f"{API_URL}/model_info", timeout=10)
        if resp.status_code == 200:
            info = resp.json()
            st.json(info.get("oof_metrics", {}))
            st.write("**Средний повторный чек:**", info.get("avg_return_amount"))
            st.write("**Количество признаков:**", len(info.get("features", [])))
            with st.expander("Список признаков"):
                st.write(info.get("features", []))
        else:
            st.warning("Модель ещё не загружена в API.")
    except Exception:
        st.info("Запустите API, чтобы увидеть метрики модели.")

with tab4:
    st.subheader("Бизнес-рекомендации по децилям")
    st.markdown(
        """
| Группа | Средний LTV | Доля (примерно) | Рекомендация |
|--------|-------------|-----------------|--------------|
| **1–3** | низкий | ~30% | Минимальные усилия: общие рассылки 1 раз в квартал |
| **4–7** | средний | ~40% | Персонализация + скидка 15% |
| **8–10** | высокий | ~30% | Максимум внимания: индивидуальные предложения, VIP-программа |

**Формула LTV**  
`pred_LTV = P(will_return) × средний повторный чек`

Модель решает задачу бинарной классификации возврата клиента (сильный дисбаланс ~1.5% позитивов).
"""
    )

st.sidebar.markdown("---")
st.sidebar.caption("Ecommerce LTV MLOps · Olist dataset")
