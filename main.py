    color="영화 유형",
    category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
    hover_name="영화 제목",
    hover_data={
        "영화 유형": True,
        "first_scrn": ":,.0f",
        "total_audi": ":,.0f",
        "days_in_top10": ":,.0f",
        "롱런 지수": ":.2f",
    },
    labels={
        x_feature: FEATURE_LABELS[x_feature],
        y_feature: FEATURE_LABELS[y_feature],
        "영화 유형": "영화 유형",
    },
    title="선택한 두 속성으로 본 영화 유형",
)
scatter_2d.update_traces(marker={"size": 8, "opacity": 0.75})
scatter_2d.update_layout(legend_title_text="영화 유형")
st.plotly_chart(scatter_2d, use_container_width=True)

st.divider()
st.subheader("3. 3차원 산점도")

if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 위에서 분석 속성을 세 개 이상 선택해 주세요.")
else:
    axis3_col1, axis3_col2, axis3_col3 = st.columns(3)
    with axis3_col1:
        x3_feature = st.selectbox(
            "X축 속성",
            options=selected_features,
            index=0,
            format_func=lambda feature: FEATURE_LABELS[feature],
            key="scatter3_x",
        )
    with axis3_col2:
        y3_feature = st.selectbox(
            "Y축 속성",
            options=selected_features,
            index=1,
            format_func=lambda feature: FEATURE_LABELS[feature],
            key="scatter3_y",
        )
    with axis3_col3:
        z3_feature = st.selectbox(
            "Z축 속성",
            options=selected_features,
            index=2,
            format_func=lambda feature: FEATURE_LABELS[feature],
            key="scatter3_z",
        )

    scatter_3d = px.scatter_3d(
        df,
        x=x3_feature,
        y=y3_feature,
        z=z3_feature,
        color="영화 유형",
        category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
        hover_name="영화 제목",
        hover_data={
            "영화 유형": True,
            "first_scrn": ":,.0f",
            "total_audi": ":,.0f",
            "days_in_top10": ":,.0f",
            "롱런 지수": ":.2f",
        },
        labels={
            x3_feature: FEATURE_LABELS[x3_feature],
            y3_feature: FEATURE_LABELS[y3_feature],
            z3_feature: FEATURE_LABELS[z3_feature],
            "영화 유형": "영화 유형",
        },
        title="세 속성으로 본 영화 유형",
    )
    scatter_3d.update_traces(marker={"size": 3, "opacity": 0.8})
    scatter_3d.update_layout(legend_title_text="영화 유형")
    st.plotly_chart(scatter_3d, use_container_width=True)

st.divider()
st.subheader("4. 유형별 편수와 속성 평균")

summary_rows = []
for label in ["㉮", "㉯", "㉰"]:
    group = df[df["영화 유형"] == label]
    summary_rows.append({
        "영화 유형": label,
        "편수": len(group),
        "스크린 수 평균": group["first_scrn"].mean(),
        "누적 관객 평균": group["total_audi"].mean(),
        "10위권 일수 평균": group["days_in_top10"].mean(),
        "롱런 지수 평균": group["롱런 지수"].mean(),
    })

summary = pd.DataFrame(summary_rows)
summary["스크린 수 평균"] = summary["스크린 수 평균"].round(1)
summary["누적 관객 평균"] = summary["누적 관객 평균"].round(0)
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].round(1)
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].round(2)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()
st.subheader("5. 유형별 누적 관객 상위 영화")

for label in ["㉮", "㉯", "㉰"]:
    group = df[df["영화 유형"] == label].sort_values(
        "total_audi", ascending=False
    ).head(5)

    st.markdown(f"**{label} 유형 — 누적 관객 상위 {len(group)}편**")
    top_movies = group[["영화 제목", "total_audi"]].rename(
        columns={
            "영화 제목": "영화 제목",
            "total_audi": "누적 관객",
        }
    ).copy()
    top_movies["누적 관객"] = top_movies["누적 관객"].map(
        lambda value: f"{value:,.0f}"
    )
    st.dataframe(top_movies, use_container_width=True, hide_index=True)

st.caption(
    "참고: K-평균은 정답 유형을 맞히는 모델이 아니라, 선택한 속성이 비슷한 영화를 "
    "세 그룹으로 묶는 비지도 학습 방법입니다. 다른 속성을 선택하면 유형이 달라질 수 있습니다."
)
