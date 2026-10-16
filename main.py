import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행·상영 특성을 바탕으로 K-평균 군집화를 해 봅니다.")

FEATURES = ["스크린 수(로그)", "누적 관객(로그)", "10위권 일수", "롱런 지수"]
RAW_MEAN_COLUMNS = {
    "스크린 수": "first_scrn",
    "누적 관객": "total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "롱런 지수",
}
FEATURE_HELP = {
    "스크린 수(로그)": "스크린 수에 상용로그(log10)를 적용한 값",
    "누적 관객(로그)": "누적 관객에 상용로그(log10)를 적용한 값",
    "10위권 일수": "박스오피스 10위권에 머문 일수",
    "롱런 지수": "누적 관객 ÷ 첫 주 관객 (최대 20)",
}


@st.cache_data(show_spinner="영화 데이터를 불러오는 중...")
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("CSV에 필요한 열이 없습니다: " + ", ".join(missing))
    return df


try:
    raw_df = load_data()
except Exception as exc:
    st.error("영화 데이터를 불러오지 못했습니다. CSV 주소와 파일 열 이름을 확인해 주세요.")
    st.code(str(exc))
    st.stop()

total_movies = len(raw_df)
df = raw_df.copy()

numeric_cols = ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 롱런 지수 계산 전에 필요한 값이 모두 있는지 확인한다.
df["롱런 지수"] = df["total_audi"] / df["first_week_audi"].where(df["first_week_audi"] != 0)
df["롱런 지수"] = df["롱런 지수"].clip(upper=20)

# 로그를 계산할 수 있도록 스크린 수와 누적 관객은 양수여야 한다.
valid_mask = (
    df["movieNm"].notna()
    & df["first_scrn"].notna()
    & df["first_week_audi"].notna()
    & df["total_audi"].notna()
    & df["days_in_top10"].notna()
    & df["롱런 지수"].notna()
    & (df["first_week_audi"] != 0)
    & (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
)
df = df.loc[valid_mask].copy()
df["스크린 수(로그)"] = df["first_scrn"].map(lambda x: math.log10(x))
df["누적 관객(로그)"] = df["total_audi"].map(lambda x: math.log10(x))

st.write(f"**전체 영화 {total_movies:,}편 · 분석에 묶은 영화 {len(df):,}편**")

if len(df) < 3:
    st.error("군집화를 하려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()

st.subheader("1. 유형을 나눌 속성 선택")
selected_features = st.multiselect(
    "군집화에 사용할 속성 (두 개 이상 선택)",
    options=FEATURES,
    default=FEATURES,
    help="선택한 속성만 표준화한 뒤 K-평균 군집화에 사용합니다.",
    format_func=lambda x: f"{x} — {FEATURE_HELP[x]}",
)
if len(selected_features) < 2:
    st.warning("군집화를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# 표준편차가 0인 속성이 있어도 표준화 결과가 안정적으로 나오도록 처리한다.
X = df[selected_features].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["군집번호"] = model.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰를 부여한다.
cluster_order = (
    df.groupby("군집번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_labels = {cluster_id: ["㉮", "㉯", "㉰"][rank] for rank, cluster_id in enumerate(cluster_order)}
df["영화 유형"] = df["군집번호"].map(cluster_labels)

st.caption("㉮·㉯·㉰는 누적 관객 평균이 높은 유형부터 붙인 이름입니다. 유형은 정답이 아니라 선택한 속성에 따른 상대적 묶음입니다.")

st.subheader("2. 2차원 산점도")
col_x, col_y = st.columns(2)
with col_x:
    x_feature = st.selectbox("가로축 속성", options=selected_features, index=0, key="scatter_x")
with col_y:
    y_default = 1 if len(selected_features) > 1 else 0
    y_feature = st.selectbox("세로축 속성", options=selected_features, index=y_default, key="scatter_y")

fig2d = px.scatter(
    df,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    hover_name="movieNm",
    hover_data={"영화 유형": True, x_feature: ":.3f", y_feature: ":.3f"},
    category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
    labels={x_feature: x_feature, y_feature: y_feature, "영화 유형": "영화 유형"},
    title="선택한 두 속성으로 본 영화 유형",
)
fig2d.update_traces(marker={"size": 8, "opacity": 0.78})
fig2d.update_layout(legend_title_text="영화 유형", margin=dict(l=20, r=20, t=55, b=20))
st.plotly_chart(fig2d, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 군집화 속성을 세 개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox("X축 속성", selected_features, index=0, key="x3")
    with c2:
        y3_index = 1 if len(selected_features) > 1 else 0
        y3 = st.selectbox("Y축 속성", selected_features, index=y3_index, key="y3")
    with c3:
        z3_index = 2 if len(selected_features) > 2 else 0
        z3 = st.selectbox("Z축 속성", selected_features, index=z3_index, key="z3")

    fig3d = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        hover_name="movieNm",
        hover_data={"영화 유형": True},
        category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
        labels={x3: x3, y3: y3, z3: z3, "영화 유형": "영화 유형"},
        title="마우스로 회전하며 살펴보는 영화 유형",
    )
    fig3d.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3d.update_layout(
        legend_title_text="영화 유형",
        margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(
            xaxis_title=x3,
            yaxis_title=y3,
            zaxis_title=z3,
        ),
    )
    st.plotly_chart(fig3d, use_container_width=True)

st.subheader("4. 유형별 편수와 속성 평균")
counts = df.groupby("영화 유형").size().rename("편수")
means = df.groupby("영화 유형").agg(
    **{
        "스크린 수 평균": ("first_scrn", "mean"),
        "누적 관객 평균": ("total_audi", "mean"),
        "10위권 일수 평균": ("days_in_top10", "mean"),
        "롱런 지수 평균": ("롱런 지수", "mean"),
    }
)
summary = counts.to_frame().join(means).reindex(["㉮", "㉯", "㉰"]).reset_index()
summary = summary.rename(columns={"영화 유형": "영화 유형"})
summary["스크린 수 평균"] = summary["스크린 수 평균"].round(1)
summary["누적 관객 평균"] = summary["누적 관객 평균"].round(0).astype("Int64")
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].round(2)
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].round(2)
st.dataframe(summary, hide_index=True, use_container_width=True)

st.caption("평균 표의 스크린 수·누적 관객은 원래 단위이며, 로그값이 아닙니다.")

st.subheader("5. 유형별 누적 관객 TOP 5")
for label in ["㉮", "㉯", "㉰"]:
    group = df[df["영화 유형"] == label].sort_values("total_audi", ascending=False).head(5)
    with st.expander(f"{label} 유형 — 누적 관객 상위 영화", expanded=True):
        if group.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            top5 = group[["movieNm", "total_audi"]].copy()
            top5 = top5.rename(columns={"movieNm": "영화 제목", "total_audi": "누적 관객"})
            top5["누적 관객"] = top5["누적 관객"].map(lambda value: f"{int(value):,}")
            top5.insert(0, "순위", range(1, len(top5) + 1))
            st.dataframe(top5, hide_index=True, use_container_width=True)

with st.expander("분석 기준 보기"):
    st.markdown(
        """
        - **스크린 수(로그):** `log10(first_scrn)`
        - **누적 관객(로그):** `log10(total_audi)`
        - **10위권 일수:** `days_in_top10` 원래 값
        - **롱런 지수:** `min(total_audi / first_week_audi, 20)`
        - 위 네 속성 중 결측값이 있거나 첫 주 관객이 0인 영화는 제외합니다. 로그 계산을 위해 스크린 수와 누적 관객이 양수인 영화만 분석합니다.
        - 선택한 속성은 `StandardScaler`로 표준화하고, `KMeans(n_clusters=3, random_state=42, n_init=10)`으로 묶습니다.
        """
    )
