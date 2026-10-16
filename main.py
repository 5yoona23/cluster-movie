X = valid[selected_features].astype(float)

# 선택한 속성을 표준화한 후 K-평균 군집화
X_scaled = StandardScaler().fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
valid["군집 원번호"] = model.fit_predict(X_scaled)

# 원래 단위 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 지정
cluster_order = (
    valid.groupby("군집 원번호")["누적 관객"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_name_map = {
    cluster_id: ["㉮", "㉯", "㉰"][i]
    for i, cluster_id in enumerate(cluster_order)
}
valid["영화 유형"] = valid["군집 원번호"].map(cluster_name_map)

st.subheader("2. 2차원 산점도")
axis_options = list(feature_map.keys())
default_x = selected_display_features[0]
default_y = selected_display_features[1]
axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_display = st.selectbox("가로축 속성", axis_options, index=axis_options.index(default_x))
with axis_col2:
    y_display = st.selectbox("세로축 속성", axis_options, index=axis_options.index(default_y))

scatter2d = valid.copy()
scatter2d["가로축 값"] = scatter2d[feature_map[x_display]]
scatter2d["세로축 값"] = scatter2d[feature_map[y_display]]
fig2d = px.scatter(
    scatter2d,
    x="가로축 값",
    y="세로축 값",
    color="영화 유형",
    hover_name="movieNm",
    hover_data={"가로축 값": ":.3f", "세로축 값": ":.3f", "영화 유형": True},
    category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
    labels={"가로축 값": x_display, "세로축 값": y_display},
    title="선택한 속성으로 살펴보는 영화 유형"
)
fig2d.update_traces(marker={"size": 8, "opacity": 0.8})
fig2d.update_layout(legend_title_text="영화 유형")
st.plotly_chart(fig2d, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_display_features) < 3:
    st.info("3차원 산점도를 보려면 위에서 군집 분석에 사용할 속성을 세 개 이상 선택하세요.")
else:
    three_defaults = selected_display_features[:3]
    c1, c2, c3 = st.columns(3)
    with c1:
        x3_display = st.selectbox("X축 속성", axis_options, index=axis_options.index(three_defaults[0]), key="x3")
    with c2:
        y3_display = st.selectbox("Y축 속성", axis_options, index=axis_options.index(three_defaults[1]), key="y3")
    with c3:
        z3_display = st.selectbox("Z축 속성", axis_options, index=axis_options.index(three_defaults[2]), key="z3")

    scatter3d = valid.copy()
    scatter3d["X 값"] = scatter3d[feature_map[x3_display]]
    scatter3d["Y 값"] = scatter3d[feature_map[y3_display]]
    scatter3d["Z 값"] = scatter3d[feature_map[z3_display]]
    fig3d = px.scatter_3d(
        scatter3d,
        x="X 값",
        y="Y 값",
        z="Z 값",
        color="영화 유형",
        hover_name="movieNm",
        hover_data={"X 값": ":.3f", "Y 값": ":.3f", "Z 값": ":.3f", "영화 유형": True},
        category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
        labels={"X 값": x3_display, "Y 값": y3_display, "Z 값": z3_display},
        title="마우스로 회전하며 살펴보는 3차원 영화 유형"
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
    롱런_지수_평균=("롱런 지수", "mean")
).reindex(["㉮", "㉯", "㉰"]).reset_index()
summary = summary.rename(columns={
    "영화 유형": "영화 유형",
    "스크린_수_평균": "스크린 수 평균",
    "누적_관객_평균": "누적 관객 평균",
    "십위권_일수_평균": "10위권 일수 평균",
    "롱런_지수_평균": "롱런 지수 평균"
})
summary["스크린 수 평균"] = summary["스크린 수 평균"].round(1)
summary["누적 관객 평균"] = summary["누적 관객 평균"].round(0).astype("Int64")
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].round(1)
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].round(2)
st.dataframe(summary, hide_index=True, use_container_width=True)

st.subheader("5. 유형별 누적 관객 상위 영화 5편")
for cluster_label in ["㉮", "㉯", "㉰"]:
    group = valid[valid["영화 유형"] == cluster_label].sort_values("누적 관객", ascending=False).head(5)
    with st.expander(f"{cluster_label} 유형 — 상위 영화 {len(group)}편", expanded=True):
        if group.empty:
            st.write("해당 유형에 영화가 없습니다.")
        else:
            top_table = group[["movieNm", "total_audi", "first_week_audi", "days_in_top10"]].rename(columns={
                "movieNm": "영화 제목",
                "total_audi": "누적 관객",
                "first_week_audi": "첫 주 관객",
                "days_in_top10": "10위권 일수"
            }).reset_index(drop=True)
            st.dataframe(top_table, hide_index=True, use_container_width=True)

st.caption("참고: K-평균은 선택한 속성으로 유형을 나눕니다. 유형 번호는 누적 관객 평균 순서에 따라 ㉮·㉯·㉰로 표시합니다.")
