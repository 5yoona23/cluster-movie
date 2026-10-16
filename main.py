    import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")
st.caption("영화 데이터를 바탕으로 비슷한 특징을 가진 영화끼리 묶어 봅니다.")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw_df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. CSV 주소가 공개되어 있고 올바른지 확인해 주세요.")
    st.code(str(e))
    st.stop()

required_columns = [
    "movieCd", "movieNm", "openDt", "genre", "nation", "first_scrn",
    "first_show", "first_date", "peak", "first_week_audi", "total_audi",
    "days_in_top10",
]
missing_columns = [col for col in required_columns if col not in raw_df.columns]
if missing_columns:
    st.error("CSV에서 필요한 열을 찾지 못했습니다: " + ", ".join(missing_columns))
    st.write("실제 열 이름:", list(raw_df.columns))
    st.stop()

# 숫자 열 변환
df = raw_df.copy()
numeric_columns = [
    "first_scrn", "first_show", "first_date", "peak", "first_week_audi",
    "total_audi", "days_in_top10",
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 분석에 사용할 네 속성(표준화 전에 단위가 큰 값은 로그 변환)
df["스크린 수"] = df["first_scrn"]
df["누적 관객"] = df["total_audi"]
df["10위권 일수"] = df["days_in_top10"]
df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"].where(df["first_week_audi"] > 0)
).clip(upper=20)

feature_original = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]
valid = df.dropna(subset=feature_original + ["movieNm"]).copy()
valid = valid[(valid["first_week_audi"] > 0) & (valid["스크린 수"] > 0) & (valid["누적 관객"] > 0)].copy()
valid["로그 스크린 수"] = valid["스크린 수"].apply(lambda x: math.log10(x))
valid["로그 누적 관객"] = valid["누적 관객"].apply(lambda x: math.log10(x))

feature_map = {
    "스크린 수 (상용로그)": "로그 스크린 수",
    "누적 관객 (상용로그)": "로그 누적 관객",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수",
}

st.write(f"**전체 영화 {len(raw_df):,}편 · 분석에 사용한 영화 {len(valid):,}편**")
if len(valid) < 3:
    st.error("유효한 영화가 3편보다 적어 군집 분석을 할 수 없습니다.")
    st.stop()

st.subheader("1. 유형을 나눌 속성과 묶음 수 선택")
selected_display_features = st.multiselect(
    "군집 분석에 사용할 속성을 두 개 이상 선택하세요.",
    options=list(feature_map.keys()),
    default=list(feature_map.keys()),
)
if len(selected_display_features) < 2:
    st.warning("군집 분석을 하려면 속성을 두 개 이상 선택해야 합니다.")
    st.stop()

max_k = min(7, len(valid) - 1)
if max_k < 2:
    st.error("묶음 수를 2 이상으로 설정하려면 분석할 영화가 더 필요합니다.")
    st.stop()

control_col, note_col = st.columns([1, 3])
with control_col:
    n_clusters = st.selectbox("묶음 수", options=list(range(2, max_k + 1)), index=min(1, max_k - 2))
with note_col:
    st.caption("묶음 수를 바꾸면 산점도, 요약표, 영화 제목 목록, 평가 그래프가 함께 다시 계산됩니다.")
    scatter2d, x="가로축 값", y="세로축 값", color="영화 유형", hover_name="movieNm",
    hover_data={"가로축 값": ":.3f", "세로축 값": ":.3f", "영화 유형": True},
    category_orders={"영화 유형": active_symbols},
    labels={"가로축 값": x_display, "세로축 값": y_display},
    title=f"{n_clusters}개 묶음으로 살펴보는 영화 유형",
)
fig2d.update_traces(marker={"size": 8, "opacity": 0.8})
fig2d.update_layout(legend_title_text="영화 유형")
st.plotly_chart(fig2d, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_display_features) < 3:
    st.info("3차원 산점도를 보려면 군집 분석에 사용할 속성을 세 개 이상 선택하세요.")
else:
    defaults3 = selected_display_features[:3]
    c1, c2, c3 = st.columns(3)
    with c1:
        x3_display = st.selectbox("X축 속성", axis_options, index=axis_options.index(defaults3[0]), key="x3")
    with c2:
        y3_display = st.selectbox("Y축 속성", axis_options, index=axis_options.index(defaults3[1]), key="y3")
    with c3:
        z3_display = st.selectbox("Z축 속성", axis_options, index=axis_options.index(defaults3[2]), key="z3")
    scatter3d = valid.copy()
    scatter3d["X 값"] = scatter3d[feature_map[x3_display]]
    scatter3d["Y 값"] = scatter3d[feature_map[y3_display]]
    scatter3d["Z 값"] = scatter3d[feature_map[z3_display]]
    fig3d = px.scatter_3d(
        scatter3d, x="X 값", y="Y 값", z="Z 값", color="영화 유형", hover_name="movieNm",
        hover_data={"X 값": ":.3f", "Y 값": ":.3f", "Z 값": ":.3f", "영화 유형": True},
        category_orders={"영화 유형": active_symbols},
        labels={"X 값": x3_display, "Y 값": y3_display, "Z 값": z3_display},
        title=f"{n_clusters}개 묶음의 3차원 분포",
    )
    fig3d.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3d.update_layout(legend_title_text="영화 유형", scene={"aspectmode": "data"})
    st.plotly_chart(fig3d, use_container_width=True)

st.subheader("4. 유형별 편수와 속성 평균")
summary = valid.groupby("영화 유형").agg(
    편수=("movieNm", "count"),
    스크린_수_평균=("스크린 수", "mean"),
    누적_관객_평균=("누적 관객", "mean"),
    십위권_일수_평균=("10위권 일수", "mean"),
    롱런_지수_평균=("롱런 지수", "mean"),
).reindex(active_symbols).reset_index()
summary = summary.rename(columns={
    "스크린_수_평균": "스크린 수 평균", "누적_관객_평균": "누적 관객 평균",
    "십위권_일수_평균": "10위권 일수 평균", "롱런_지수_평균": "롱런 지수 평균",
})
summary["스크린 수 평균"] = summary["스크린 수 평균"].round(1)
summary["누적 관객 평균"] = summary["누적 관객 평균"].round(0).astype("Int64")
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].round(1)
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].round(2)
st.dataframe(summary, hide_index=True, use_container_width=True)

st.subheader("5. 유형별 누적 관객 상위 영화 5편")
for symbol in active_symbols:
    group = valid[valid["영화 유형"] == symbol].sort_values("누적 관객", ascending=False).head(5)
    with st.expander(f"{symbol} 유형 — 상위 영화 {len(group)}편", expanded=True):
        top_table = group[["movieNm", "total_audi", "first_week_audi", "days_in_top10"]].rename(columns={
            "movieNm": "영화 제목", "total_audi": "누적 관객",
            "first_week_audi": "첫 주 관객", "days_in_top10": "10위권 일수",
        }).reset_index(drop=True)
        st.dataframe(top_table, hide_index=True, use_container_width=True)

st.subheader("6. 묶음 수에 따른 중심 거리 제곱합")
st.write("선택한 속성을 그대로 사용해 묶음 수를 1부터 7까지 바꾸며, 각 영화와 자기 묶음 중심 사이 거리의 제곱합을 계산합니다.")
max_eval_k = min(7, len(valid))
inertia_rows = []
inertia_values = []
for k in range(1, max_eval_k + 1):
    eval_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    eval_model.fit(X_scaled)
    inertia_values.append(float(eval_model.inertia_))
    inertia_rows.append({"묶음 수": k, "중심 거리 제곱합": float(eval_model.inertia_)})

fig_inertia = go.Figure()
fig_inertia.add_trace(go.Scatter(
    x=list(range(1, max_eval_k + 1)), y=inertia_values,
    mode="lines+markers", name="중심 거리 제곱합",
))
fig_inertia.add_vline(x=n_clusters, line_dash="dash", line_color="red", annotation_text=f"현재 k={n_clusters}")
fig_inertia.update_layout(
    title="묶음 수와 중심 거리 제곱합", xaxis_title="묶음 수(k)",
    yaxis_title="중심 거리 제곱합", xaxis={"dtick": 1},
)
st.plotly_chart(fig_inertia, use_container_width=True)

inertia_table = pd.DataFrame(inertia_rows)
inertia_table["직전 값보다 줄어든 양"] = inertia_table["중심 거리 제곱합"].shift(1) - inertia_table["중심 거리 제곱합"]
inertia_table["중심 거리 제곱합"] = inertia_table["중심 거리 제곱합"].map(lambda v: f"{v:,.3f}")
inertia_table["직전 값보다 줄어든 양"] = inertia_table["직전 값보다 줄어든 양"].map(lambda v: "" if pd.isna(v) else f"{v:,.3f}")
st.dataframe(inertia_table, hide_index=True, use_container_width=True)

st.subheader("7. 현재 묶음의 실루엣 점수")
if 2 <= n_clusters < len(valid) and valid["군집 원번호"].nunique() >= 2:
    try:
        score = silhouette_score(X_scaled, valid["군집 원번호"])
        st.write(f"현재 묶음 수 **{n_clusters}개**의 실루엣 점수는 **{score:.3f}**입니다. (가능한 범위: -1~1, 1에 가까울수록 묶음이 뚜렷합니다.)")
    except Exception as e:
        st.warning("실루엣 점수를 계산하지 못했습니다.")
        st.caption(str(e))
else:
    st.info("실루엣 점수를 계산하려면 두 개 이상의 묶음과 충분한 데이터가 필요합니다.")

st.caption("참고: 영화 유형 기호는 각 묶음의 평균 누적 관객이 많은 순서로 ㉮부터 붙입니다. 군집화와 평가는 위에서 선택한 속성을 사용합니다.")
