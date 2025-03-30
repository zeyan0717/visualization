# Advanced Interactive Nutrition Dashboard
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Load and clean dataset
df_raw = pd.read_csv("nutrition.csv")
df = df_raw.copy()

# Clean numeric fields
def extract_numeric(val):
    try:
        return float(str(val).split()[0].replace(',', ''))
    except:
        return None

columns_to_convert = ['calories', 'protein', 'carbohydrate', 'sugars', 'sodium', 'fiber', 'fat', 'water']
for col in columns_to_convert:
    df[col] = df[col].apply(extract_numeric)

# Drop nulls for core fields
df.dropna(subset=['calories', 'protein', 'carbohydrate', 'sugars'], inplace=True)

# Compute health score
score_df = df[['calories', 'protein', 'fiber', 'fat', 'sugars', 'sodium']].copy()
scaler = MinMaxScaler()
score_scaled = pd.DataFrame(scaler.fit_transform(score_df), columns=score_df.columns)
df['health_score'] = (
    score_scaled['protein'] * 2 +
    score_scaled['fiber'] * 1.5 -
    score_scaled['fat'] -
    score_scaled['sugars'] -
    score_scaled['sodium'] -
    score_scaled['calories']
)

# Theme toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown("""<style>body { background-color: #1e1e1e; color: white; }</style>""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔎 Filter Options")
cal_range = st.sidebar.slider("Calories Range", 0, int(df['calories'].max()), (0, 500))
filtered_df = df[(df['calories'] >= cal_range[0]) & (df['calories'] <= cal_range[1])]

nutrients = ['protein', 'carbohydrate', 'sugars', 'fat', 'fiber', 'water', 'sodium']

# Tabs layout
tabs = st.tabs(["📊 Nutrient Analysis", "🧭 Radar Chart", "🌡️ Heatmap", "🏆 Health Ranking", "🤖 AI Recommender"])

with tabs[0]:
    st.header("📊 Nutrient Averages")
    selected_nutrients = st.multiselect("Select Nutrients", nutrients, default=['protein', 'sugars'])
    if selected_nutrients:
        avg_vals = filtered_df[selected_nutrients].mean().sort_values(ascending=False)
        fig_avg = px.bar(x=avg_vals.index, y=avg_vals.values, labels={'x': 'Nutrient', 'y': 'Avg per 100g'})
        st.plotly_chart(fig_avg)

    st.subheader("📈 Scatter Plot")
    scatter_y = st.selectbox("Y-axis Nutrient", selected_nutrients)
    fig_scatter = px.scatter(filtered_df, x='calories', y=scatter_y, hover_name='name')
    st.plotly_chart(fig_scatter)

with tabs[1]:
    st.header("🧭 Nutrient Radar")
    food_choice = st.selectbox("Choose a Food", filtered_df['name'].dropna().unique())
    nutrient_vals = filtered_df[filtered_df['name'] == food_choice][selected_nutrients].iloc[0]
    fig_radar = go.Figure(go.Scatterpolar(r=nutrient_vals.values, theta=selected_nutrients, fill='toself'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    st.plotly_chart(fig_radar)

with tabs[2]:
    st.header("🌡️ Nutrient Correlation Heatmap")
    if len(selected_nutrients) >= 2:
        fig_heat = px.imshow(filtered_df[selected_nutrients].corr(), text_auto=True, aspect='auto')
        st.plotly_chart(fig_heat)

with tabs[3]:
    st.header("🏆 Top 10 Healthiest Foods")
    top_health = filtered_df[['name', 'health_score']].sort_values(by='health_score', ascending=False).head(10)
    fig_health = px.bar(top_health, x='name', y='health_score', labels={'health_score': 'Health Score'})
    st.plotly_chart(fig_health)

with tabs[4]:
    st.header("🤖 AI Nutrition Recommender")
    goal = st.radio("Your goal:", ["增肌 (High Protein)", "减脂 (Low Fat/Sugar)", "控糖 (Low Sugar)", "高纤维饮食 (High Fiber)"])
    if goal == "增肌 (High Protein)":
        recs = filtered_df.sort_values(by='protein', ascending=False).head(10)
    elif goal == "减脂 (Low Fat/Sugar)":
        recs = filtered_df[(filtered_df['fat'] < 5) & (filtered_df['sugars'] < 5)].sort_values(by='health_score', ascending=False).head(10)
    elif goal == "控糖 (Low Sugar)":
        recs = filtered_df[filtered_df['sugars'] < 2].sort_values(by='health_score', ascending=False).head(10)
    else:
        recs = filtered_df[filtered_df['fiber'] > 5].sort_values(by='health_score', ascending=False).head(10)

    st.markdown("**Top Recommendations:**")
    st.table(recs[['name', 'protein', 'fiber', 'fat', 'sugars', 'calories']])

# Footer
st.markdown("---")
st.markdown("Data Source: [Kaggle - Nutritional Values for Common Foods](https://www.kaggle.com/datasets/trolukovich/nutritional-values-for-common-foods-and-products)")
