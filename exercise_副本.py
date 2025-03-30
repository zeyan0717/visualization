import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ========== 1. 页面基本设置 ==========
st.set_page_config(
    page_title="Nutrition Advanced Dashboard",
    layout="wide"
)
st.title("营养数据可视化与分析 - 进阶示例")

# ========== 2. 数据读取与缓存 ==========
@st.cache_data  # 当数据源不变时，可使用缓存加速
def load_data():
    # 方式1：从 GitHub raw 链接读取
    # url = "https://raw.githubusercontent.com/<你的GitHub用户名>/<仓库名>/main/nutrition.csv"
    # df = pd.read_csv(url)

    # 方式2：从本地文件读取（在 Streamlit Cloud 上，需要在相同仓库中）
    df = pd.read_csv("./nutrition.csv")
    return df

df_raw = load_data()

# ========== 3. 数据清洗与预处理 ==========
df = df_raw.copy()
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# 让我们假设数据具有以下列：
# ["Food", "Category", "Calories", "Fat", "Protein", "Carbs"]
# 具体以你的 CSV 实际列为准，若不一致，请改下列名
# df.rename(columns={"旧列名": "新列名"}, inplace=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ========== 4. 侧边栏：多重筛选控件 ==========
st.sidebar.header("筛选条件")

# 4.1 按 Category 多选
if "Category" in df.columns:
    all_categories = sorted(df["Category"].dropna().unique().tolist())
    category_selected = st.sidebar.multiselect(
        "选择类别（可多选）",
        options=all_categories,
        default=all_categories
    )
else:
    category_selected = []

# 4.2 Text Input：按食品关键词模糊搜索
food_keyword = st.sidebar.text_input("输入 Food 关键词 (可留空)", "")

# 4.3 针对某些数值列，增加区间过滤（示例用 Calories）
if "Calories" in df.columns:
    cal_min, cal_max = int(df["Calories"].min()), int(df["Calories"].max())
    calories_range = st.sidebar.slider(
        "卡路里范围筛选",
        min_value=cal_min,
        max_value=cal_max,
        value=(cal_min, cal_max)
    )
else:
    calories_range = (None, None)

# 用一个函数封装过滤逻辑
def filter_data(dataframe):
    temp = dataframe.copy()
    # 按类别多选过滤
    if "Category" in temp.columns and category_selected:
        temp = temp[temp["Category"].isin(category_selected)]
    # 按食品关键词过滤
    if food_keyword:
        temp = temp[temp["Food"].str.contains(food_keyword, case=False)]
    # 按卡路里区间过滤
    if "Calories" in temp.columns and calories_range != (None, None):
        low, high = calories_range
        temp = temp[(temp["Calories"] >= low) & (temp["Calories"] <= high)]
    return temp

df_filtered = filter_data(df)

# ========== 5. 多页面/多标签布局 ==========
# Streamlit 新版支持 st.tabs()，也可以用 radio/selectbox 来切换页面
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["数据概览", "交互式图表", "相关性分析", "分组聚合", "简单ML示例"]
)

# ========== Tab1: 数据概览 ==========
with tab1:
    st.subheader("原始数据 (前 10 行)")
    st.dataframe(df.head(10))

    st.subheader("描述性统计（所有数值列）")
    st.write(df.describe())

    st.markdown("---")
    st.warning(f"当前筛选后的数据量: {len(df_filtered)} 行")
    st.write("筛选后的数据 (前 20 行)：")
    st.dataframe(df_filtered.head(20))

# ========== Tab2: 交互式图表 ==========
with tab2:
    st.subheader("柱状图: Food vs Calories")
    if df_filtered.empty:
        st.error("筛选后无数据，无法绘制图表。")
    else:
        if "Food" in df_filtered.columns and "Calories" in df_filtered.columns:
            fig_bar = px.bar(
                df_filtered,
                x="Food",
                y="Calories",
                color="Category" if "Category" in df_filtered.columns else None,
                title="不同食品的热量对比 (可滚动查看)",
            )
            fig_bar.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("缺少 'Food' 或 'Calories' 列，无法生成柱状图。")

    st.subheader("箱线图: 分布可视化 (Fat、Protein、Carbs)")
    numeric_options = ["Fat", "Protein", "Carbs"]
    numeric_options = [col for col in numeric_options if col in df_filtered.columns]
    if not numeric_options or df_filtered.empty:
        st.info("数据列不足，或筛选后无数据。")
    else:
        # 使用 Plotly 画箱线图，可一次对多列做对比
        fig_box = go.Figure()
        for col in numeric_options:
            fig_box.add_trace(go.Box(y=df_filtered[col], name=col))
        fig_box.update_layout(title="营养数据箱线图")
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("散点矩阵: 多变量关系 (数值列)")
    if len(numeric_cols) >= 2 and not df_filtered.empty:
        fig_scatter_matrix = px.scatter_matrix(
            df_filtered,
            dimensions=numeric_cols,  # 所有数值列
            color="Category" if "Category" in df_filtered.columns else None,
            title="数值列散点矩阵"
        )
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)
    else:
        st.info("数值列不足，或筛选后无数据。")

# ========== Tab3: 相关性分析 ==========
with tab3:
    st.subheader("数值列相关性热力图")
    if len(numeric_cols) < 2:
        st.error("数值列太少，无法计算相关性。")
    else:
        corr_matrix = df_filtered[numeric_cols].corr()
        # 用 Plotly Express 画热力图
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="相关系数热力图 (Pearson)"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ========== Tab4: 分组聚合 (Pivot Table) ==========
with tab4:
    st.subheader("分组汇总统计")
    if "Category" not in df_filtered.columns:
        st.info("没有 'Category' 列，无法示例分组。")
    else:
        # 让用户选择数值列、聚合方式
        agg_col = st.selectbox("选择要聚合的数值列", numeric_cols)
        agg_func = st.selectbox("选择聚合方式", ["mean", "sum", "max", "min", "count"])

        # 根据筛选后的 df 做分组
        if df_filtered.empty:
            st.error("筛选后无数据，无法分组聚合。")
        else:
            grouped = df_filtered.groupby("Category").agg({agg_col: agg_func})
            grouped.columns = [f"{agg_col}_{agg_func}"]
            st.dataframe(grouped)

            st.subheader("分组结果可视化")
            fig_group = px.bar(
                grouped.reset_index(),
                x="Category",
                y=f"{agg_col}_{agg_func}",
                title=f"按Category分组的 {agg_col} - {agg_func}",
            )
            st.plotly_chart(fig_group, use_container_width=True)

# ========== Tab5: 简单ML示例 (线性回归) ==========
with tab5:
    st.subheader("线性回归示例: 预测 Protein (可自行修改目标列)")
    # 这里假设要预测 Protein，使用 Fat、Carbs、Calories 等做特征
    # 根据实际数据情况做更改

    # 首先选出数值列，不含目标列
    possible_features = [col for col in numeric_cols if col != "Protein"]
    if "Protein" not in df_filtered.columns:
        st.warning("数据中没有 'Protein' 列，无法做此演示。")
    elif len(possible_features) == 0:
        st.warning("除了 Protein，没有其他数值列，无法做回归。")
    else:
        selected_features = st.multiselect(
            "选择特征列 (Feature columns)",
            options=possible_features,
            default=possible_features
        )
        # 当选了至少一个特征时，进行训练
        if len(selected_features) > 0 and not df_filtered.empty:
            X = df_filtered[selected_features]
            y = df_filtered["Protein"]

            # 拆分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 模型训练
            model = LinearRegression()
            model.fit(X_train, y_train)

            # 模型评价
            score = model.score(X_test, y_test)

            st.write("模型训练完成！")
            st.write("所用特征：", selected_features)
            st.write("R^2 (测试集) =", round(score, 4))

            # 简单预测可视化（如果只选1个特征时，可以做散点+回归线）
            if len(selected_features) == 1:
                feat = selected_features[0]
                y_pred = model.predict(X_test[[feat]])
                chart_data = pd.DataFrame({
                    feat: X_test[feat],
                    "Real_Protein": y_test,
                    "Pred_Protein": y_pred
                })
                fig_reg = px.scatter(
                    chart_data,
                    x=feat, y="Real_Protein",
                    title=f"实际值 vs. 预测值 (特征: {feat})"
                )
                # 添加预测线
                fig_reg.add_trace(
                    go.Scatter(
                        x=chart_data[feat],
                        y=chart_data["Pred_Protein"],
                        mode='markers',
                        name='Predicted',
                        marker=dict(symbol='x', size=8)
                    )
                )
                st.plotly_chart(fig_reg, use_container_width=True)
            else:
                st.info("选择了多个特征，暂不做可视化回归线。可自行在上方只选 1 个特征来查看散点回归图。")

        else:
            st.info("请在左侧或上方选择特征列，或检查数据是否为空。")

st.markdown("---")
