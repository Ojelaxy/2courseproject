from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Кластеризация российских вузов",
    layout="wide"
)

DATA_FILE_CANDIDATES = [
    "universities_clustered_2025.csv",
]
PROFILE_FILE_CANDIDATES = [
    "cluster_profile_2025.csv", 
]
REPRESENTATIVES_FILE_CANDIDATES = [
    "cluster_representatives_2025.csv",
]

FEATURE_COLS = [
    "ege_avg",
    "students_count",
    "foreign_share",
    "pps_salary",
    "publications_per_staff",
    "citations_per_staff",
    "income_per_student",
]

FEATURE_LABELS = {
    "ege_avg": "Средний ЕГЭ",
    "students_count": "Приведённый контингент студентов",
    "foreign_share": "Доля иностранных студентов, %",
    "pps_salary": "Средняя зарплата ППС, тыс. руб.",
    "publications_per_staff": "Публикации на 1 НПР",
    "citations_per_staff": "Цитирования на 1 НПР",
    "income_per_student": "Доходы на 1 студента, тыс. руб.",
}

CLUSTER_TITLES = {
    0: "Небольшие вузы с более низкими показателями",
    1: "Сильные крупные вузы",
    2: "Вузы со средними устойчивыми показателями",
}

CLUSTER_DESCRIPTIONS = {
    0: (
        "В этот кластер попали сравнительно небольшие вузы, для которых характерны "
        "более низкие значения среднего ЕГЭ, доли иностранных студентов, средней "
        "зарплаты ППС и научных показателей."
    ),
    1: (
        "В этот кластер попали крупные и сильные вузы с высокими значениями среднего "
        "ЕГЭ, международной активности, зарплаты ППС и научной результативности."
    ),
    2: (
        "В этот кластер попали вузы со средними и устойчивыми показателями. Они занимают "
        "промежуточное положение между первым и нулевым кластерами по большинству признаков."
    ),
}


def resolve_existing_file(candidates: list[str]):
    for file_name in candidates:
        if Path(file_name).exists():
            return file_name
    return None


def validate_dataframe(df: pd.DataFrame, required_columns: list[str], df_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"В файле {df_name} отсутствуют обязательные столбцы: {', '.join(missing)}"
        )


@st.cache_data
def load_data():
    data_file = resolve_existing_file(DATA_FILE_CANDIDATES)
    profile_file = resolve_existing_file(PROFILE_FILE_CANDIDATES)
    representatives_file = resolve_existing_file(REPRESENTATIVES_FILE_CANDIDATES)

    if data_file is None:
        raise FileNotFoundError(
            f"Не найден ни один из файлов с данными вузов: {', '.join(DATA_FILE_CANDIDATES)}"
        )
    if profile_file is None:
        raise FileNotFoundError(
            f"Не найден ни один из файлов с профилями кластеров: {', '.join(PROFILE_FILE_CANDIDATES)}"
        )

    df = pd.read_csv(data_file, sep=";")
    profile = pd.read_csv(profile_file, sep=";", index_col=0)

    representatives = None
    if representatives_file is not None:
        representatives = pd.read_csv(representatives_file, sep=";")

    required_df_columns = ["id", "name", "region", "cluster", "pca1", "pca2"] + FEATURE_COLS
    validate_dataframe(df, required_df_columns, data_file)

    df["name"] = df["name"].astype(str).str.strip()
    df["region"] = df["region"].fillna("Не указан").astype(str).str.strip()

    for col in ["id", "cluster"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["id", "cluster", "name"]).copy()
    df["id"] = df["id"].astype(int)
    df["cluster"] = df["cluster"].astype(int)

    for col in FEATURE_COLS + ["pca1", "pca2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cluster" in profile.columns:
        profile["cluster"] = pd.to_numeric(profile["cluster"], errors="coerce")
        profile = profile.dropna(subset=["cluster"]).copy()
        profile["cluster"] = profile["cluster"].astype(int)
        profile = profile.set_index("cluster")
    else:
        profile.index = pd.to_numeric(profile.index, errors="coerce")
        profile = profile.dropna(axis=0).copy()
        profile.index = profile.index.astype(int)

    for col in FEATURE_COLS:
        profile[col] = pd.to_numeric(profile[col], errors="coerce")

    if representatives is not None:
        reps_required = ["cluster", "name", "region"]
        validate_dataframe(representatives, reps_required, representatives_file)
        representatives["cluster"] = pd.to_numeric(representatives["cluster"], errors="coerce")
        representatives = representatives.dropna(subset=["cluster", "name"]).copy()
        representatives["cluster"] = representatives["cluster"].astype(int)
        representatives["name"] = representatives["name"].astype(str).str.strip()
        representatives["region"] = representatives["region"].fillna("Не указан").astype(str).str.strip()

    return df, profile, representatives, data_file, profile_file, representatives_file


def format_cluster_option(cluster_id: int) -> str:
    title = CLUSTER_TITLES.get(cluster_id, f"Кластер {cluster_id}")
    return f"{cluster_id} — {title}"


def parse_cluster_option(option: str):
    if option == "Кластер не выбран":
        return None
    return int(option.split("—")[0].strip())


def get_similar_universities(df: pd.DataFrame, selected_row: pd.Series, top_n: int = 10) -> pd.DataFrame:
    same_cluster = df[df["cluster"] == selected_row["cluster"]].copy()

    if len(same_cluster) <= 1:
        return same_cluster

    features = same_cluster[FEATURE_COLS].copy()
    features = (features - features.mean()) / features.std(ddof=0)
    features = features.fillna(0)

    selected_index = selected_row.name
    selected_vector = features.loc[selected_index]

    distances = np.sqrt(((features - selected_vector) ** 2).sum(axis=1))
    same_cluster["similarity_distance"] = distances

    result = (
        same_cluster
        .sort_values("similarity_distance")
        .drop(index=selected_index, errors="ignore")
        .head(top_n)
    )
    return result


def draw_pca_plot(df: pd.DataFrame, selected_id=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["pca1"], df["pca2"], c=df["cluster"], alpha=0.6)

    if selected_id is not None:
        selected = df[df["id"] == selected_id]
        if not selected.empty:
            ax.scatter(selected["pca1"], selected["pca2"], s=180, marker="X")

    ax.set_title("Распределение вузов по кластерам (PCA)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True)
    return fig


def draw_compare_chart(selected_row: pd.Series, cluster_mean: pd.Series):
    labels = [FEATURE_LABELS[col] for col in FEATURE_COLS]
    selected_values = [selected_row[col] for col in FEATURE_COLS]
    cluster_values = [cluster_mean[col] for col in FEATURE_COLS]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, selected_values, width, label="Выбранный вуз")
    ax.bar(x + width / 2, cluster_values, width, label="Среднее по кластеру")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Сравнение вуза со средним профилем кластера")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    return fig


def make_value_table(row: pd.Series, value_column: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Показатель": [FEATURE_LABELS[col] for col in FEATURE_COLS],
        value_column: [row[col] for col in FEATURE_COLS],
    })


def show_cluster_report(df: pd.DataFrame, cluster_profile: pd.DataFrame, cluster_id: int, representatives: pd.DataFrame | None):
    st.subheader("Отчёт по выбранному кластеру")
    st.write(f"**Кластер:** {cluster_id} — {CLUSTER_TITLES.get(cluster_id, 'Без названия')}")
    st.write("**Что означает этот кластер:**", CLUSTER_DESCRIPTIONS.get(cluster_id, "Описание отсутствует"))

    cluster_df = df[df["cluster"] == cluster_id].copy()
    st.write(f"**Количество вузов в кластере:** {len(cluster_df)}")

    if cluster_id in cluster_profile.index:
        st.write("### Средние значения показателей по кластеру")
        cluster_table = make_value_table(cluster_profile.loc[cluster_id], "Среднее по кластеру")
        st.dataframe(cluster_table, use_container_width=True)

    if representatives is not None:
        reps = representatives[representatives["cluster"] == cluster_id].copy()
        if not reps.empty:
            st.write("### Типичные представители кластера")
            show_cols = [col for col in ["name", "region", "distance_to_center"] if col in reps.columns]
            st.dataframe(reps[show_cols].reset_index(drop=True), use_container_width=True)

    st.write("### Все вузы данного кластера")
    show_cols = ["name", "region"] + FEATURE_COLS
    st.dataframe(
        cluster_df[show_cols].sort_values(["region", "name"]).reset_index(drop=True),
        use_container_width=True,
    )


try:
    df, cluster_profile, representatives, data_file_name, profile_file_name, representatives_file_name = load_data()
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()

st.title("Анализ и кластеризация российских вузов по показателям эффективности")
st.write("Приложение для анализа вузов по результатам кластеризации.")

with st.expander("Какие файлы загружены"):
    st.write(f"**Файл с данными вузов:** {data_file_name}")
    st.write(f"**Файл со средними профилями кластеров:** {profile_file_name}")
    if representatives_file_name is not None:
        st.write(f"**Файл с типичными представителями кластеров:** {representatives_file_name}")
    else:
        st.write("**Файл с типичными представителями кластеров:** не найден, этот блок необязателен")

with st.expander("Что означают кластеры"):
    for cluster_id in sorted(df["cluster"].unique()):
        st.markdown(
            f"**{cluster_id} — {CLUSTER_TITLES.get(cluster_id, f'Кластер {cluster_id}')}**\n\n"
            f"{CLUSTER_DESCRIPTIONS.get(cluster_id, 'Описание отсутствует')}"
        )

regions = ["Регион не выбран"] + sorted(df["region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Выберите регион", regions)

cluster_options = ["Кластер не выбран"] + [format_cluster_option(x) for x in sorted(df["cluster"].unique())]
selected_cluster_option = st.sidebar.selectbox("Выберите кластер", cluster_options)
selected_cluster = parse_cluster_option(selected_cluster_option)

filtered_df = df.copy()
if selected_region != "Регион не выбран":
    filtered_df = filtered_df[filtered_df["region"] == selected_region].copy()

if selected_cluster is not None:
    filtered_df = filtered_df[filtered_df["cluster"] == selected_cluster].copy()

if filtered_df.empty:
    st.warning("По выбранным фильтрам данные не найдены. Измените регион или кластер.")
    st.stop()

university_names = ["Вуз не выбран"] + sorted(filtered_df["name"].dropna().unique().tolist())
selected_name = st.sidebar.selectbox("Выберите вуз", university_names)

if selected_cluster is not None:
    show_cluster_report(df, cluster_profile, selected_cluster, representatives)

st.divider()

if selected_name == "Вуз не выбран":
    st.info("Выберите вуз, чтобы посмотреть его подробную карточку.")
    fig_pca = draw_pca_plot(df, None)
    st.pyplot(fig_pca)
    st.stop()

selected_row_df = filtered_df[filtered_df["name"] == selected_name].head(1)
if selected_row_df.empty:
    st.warning("Не удалось найти выбранный вуз. Попробуйте выбрать другой вариант.")
    st.stop()

selected_row = selected_row_df.iloc[0]
cluster_id = int(selected_row["cluster"])

st.subheader("Общая информация о вузе")
col1, col2, col3 = st.columns(3)
col1.metric("Кластер", cluster_id)
col2.metric("Регион", selected_row["region"])
col3.metric("ID вуза", int(selected_row["id"]))

st.write("**Название кластера:**", CLUSTER_TITLES.get(cluster_id, f"Кластер {cluster_id}"))
st.write("**Описание кластера:**", CLUSTER_DESCRIPTIONS.get(cluster_id, "Описание отсутствует"))

st.write("### Показатели выбранного вуза")
selected_table = make_value_table(selected_row, "Значение")
st.dataframe(selected_table, use_container_width=True)

if cluster_id not in cluster_profile.index:
    st.error("Для выбранного кластера не найден профиль.")
    st.stop()

st.write("### Средний профиль кластера")
cluster_mean = cluster_profile.loc[cluster_id]
cluster_table = make_value_table(cluster_mean, "Среднее по кластеру")
st.dataframe(cluster_table, use_container_width=True)

st.write("### Сравнение выбранного вуза со средним профилем кластера")
fig_compare = draw_compare_chart(selected_row, cluster_mean)
st.pyplot(fig_compare)

st.write("### Похожие вузы из того же кластера")
similar_df = get_similar_universities(df, selected_row, top_n=10)

if not similar_df.empty:
    st.dataframe(
        similar_df[["name", "region", "cluster"] + FEATURE_COLS],
        use_container_width=True,
    )
else:
    st.info("Для этого вуза не удалось подобрать похожие организации.")

st.write("### Визуализация кластеров")
fig_pca = draw_pca_plot(df, int(selected_row["id"]))
st.pyplot(fig_pca)
