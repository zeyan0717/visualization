import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# 数据加载和清洗函数
@st.cache_data
def load_and_clean_data():
    df_raw = pd.read_csv("nutrition.csv")
    df = df_raw.copy()

    # 定义需要转换的列
    columns_to_convert = [
        'calories', 'protein', 'carbohydrate', 'sugars', 'sodium',
        'fiber', 'fat', 'water'
    ]

    # 将字段转换为数值型
    def extract_numeric(val):
        try:
            return float(str(val).split()[0].replace(',', ''))
        except:
            return None

    for col in columns_to_convert:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)

    # 删除缺失核心营养字段的行
    df.dropna(subset=['calories', 'protein', 'carbohydrate', 'sugars'], inplace=True)

    # 计算健康评分
    scaler = MinMaxScaler()
    score_df = df[['calories', 'protein', 'fiber', 'fat', 'sugars', 'sodium']].copy()
    score_df_scaled = pd.DataFrame(scaler.fit_transform(score_df), columns=score_df.columns)
    df['health_score'] = (
        score_df_scaled['protein'] * 2 +
        score_df_scaled['fiber'] * 1.5 -
        score_df_scaled['fat'] -
        score_df_scaled['sugars'] -
        score_df_scaled['sodium'] -
        score_df_scaled['calories']
    )

    return df

# 加载数据
df = load_and_clean_data()

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio("选择页面", ["营养分析", "营养雷达", "热力图", "健康排行", "AI 推荐"])

# 侧边栏筛选器
st.sidebar.header("筛选选项")
cal_range = st.sidebar.slider("卡路里范围", 0, int(df['calories'].max()), (0, 500))
selected_nutrients = st.sidebar.multiselect(
    "选择营养素",
    ['protein', 'carbohydrate', 'sugars', 'fat', 'fiber', 'water', 'sodium'],
    default=['protein', 'sugars']
)

# 应用卡路里范围筛选
filtered_df = df[(df['calories'] >= cal_range[0]) & (df['calories'] <= cal_range[1])]

# 页面内容
if page == "营养分析":
    st.title("🥗 营养分析")
    if selected_nutrients:
        avg_vals = filtered_df[selected_nutrients].mean().sort_values(ascending=False)
        fig_avg = px.bar(
            x=avg_vals.index,
            y=avg_vals.values,
            labels={'x': '营养素', 'y': '每100g的平均含量'}
        )
        st.plotly_chart(fig_avg)

elif page == "营养雷达":
    st.title("🧭 营养雷达")
    food_option = st.selectbox("选择食物", filtered_df['name'].dropna().unique())
    if food_option:
        food_row = filtered_df[filtered_df['name'] == food_option][selected_nutrients].iloc[0]
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=food_row.values,
            theta=selected_nutrients,
            fill='toself',
            name=food_option
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        st.plotly_chart(fig_radar)

elif page == "热力图":
    st.title("🌡️ 营养素相关性热力图")
    if len(selected_nutrients) >= 2:
        fig_heat = px.imshow(filtered_df[selected_nutrients].corr(), text_auto=True, aspect="auto")
        st.plotly_chart(fig_heat)

elif page == "健康排行":
    st.title("🏆 健康评分排行")
    top_health = filtered_df[['name', 'health_score']].sort_values(by='health_score', ascending=False).head(10)
    fig_health = px.bar(top_health, x='name', y='health_score', labels={'health_score': '健康评分'})
    st.plotly_chart(fig_health)

elif page == "AI 推荐":
    st.title("🤖 AI 营养推荐")
    goal = st.radio("你的目标：", ["增肌", "减脂", "控糖", "高纤维饮食"])
    if goal == "增肌":
        recs = filtered_df.sort_values(by='protein', ascending=False).head(10)
    elif goal == "减脂":
        recs = filtered_df[(filtered_df['fat'] < 5) & (filtered_df['sugars'] < 5)].sort_values(by='health_score', ascending=False).head(10)
    elif goal == "控糖":
        recs = filtered_df[filtered_df['sugars'] < 2].sort_values(by='health_score', ascending=False).head(10)
    else:
        recs = filtered_df[filtered_df['fiber'] > 5].sort_values(by='health_score', ascending=False).head(10)

    st.markdown("**推荐食物：**")
    st.table(recs[['name', 'protein', 'fiber', 'fat', 'sugars', 'calories']])

# 页脚
st.markdown("---")
st.markdown("数据来源：[Kaggle - 常见食物的营养价值](https://www.kaggle.com/datasets/trolukovich/nutritional-values-for-common-foods-and-products)")
