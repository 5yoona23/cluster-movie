
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
