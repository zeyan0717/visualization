# Interactive nutrition dashboard using Streamlit
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Load and clean dataset
df_raw = pd.read_csv("nutrition.csv")
df = df_raw.copy()

# Define columns to clean
columns_to_convert = [
    'calories', 'protein', 'carbohydrate', 'sugars', 'sodium',
    'fiber', 'fat', 'water'
]

# Convert fields like "12.4 g" or "6 mg" to numeric
def extract_numeric(val):
    try:
        return float(str(val).split()[0].replace(',', ''))
    except:
        return None

for col in columns_to_convert:
    if col in df.columns:
        df[col] = df[col].apply(extract_numeric)

# Drop rows missing core nutritional fields
df.dropna(subset=['calories', 'protein', 'carbohydrate', 'sugars'], inplace=True)

# Add a simple health score: higher protein and fiber, lower fat, sugar, sodium, and calories
# Normalize fields between 0-1, then weight them
from sklearn.preprocessing import MinMaxScaler
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

# Sidebar filters
st.sidebar.header("📊 Filter Options")
cal_range = st.sidebar.slider("Calories Range", 0, int(df['calories'].max()), (0, 500))
selected_nutrients = st.sidebar.multiselect(
    "Compare Nutrients",
    ['protein', 'carbohydrate', 'sugars', 'fat', 'fiber', 'water', 'sodium'],
    default=['protein', 'sugars']
)

# Apply calorie range filter
filtered_df = df[(df['calories'] >= cal_range[0]) & (df['calories'] <= cal_range[1])]

# Title
st.title("🥗 Interactive Nutrition Dashboard")
st.markdown("""
Explore nutritional profiles of nearly 9,000 food items. Use filters to focus on specific calorie ranges and compare nutrients interactively.
""")

# Section 1: Nutrient Averages
st.subheader("🔍 Average Nutrient Content")
if selected_nutrients:
    avg_vals = filtered_df[selected_nutrients].mean().sort_values(ascending=False)
    fig_avg = px.bar(
        x=avg_vals.index,
        y=avg_vals.values,
        labels={'x': 'Nutrient', 'y': 'Average per 100g'}
    )
    st.plotly_chart(fig_avg)

# Section 2: Top 10 High-Protein Foods
st.subheader("🏆 Top 10 High-Protein Foods")
top_protein = filtered_df[['name', 'protein']].dropna().sort_values(by='protein', ascending=False).head(10)
fig_top = px.bar(top_protein, x='name', y='protein', labels={'protein': 'Protein (g)'})
st.plotly_chart(fig_top)

# Section 3: Scatter Plot
st.subheader("📈 Scatter: Calories vs Nutrient")
if selected_nutrients:
    scatter_choice = st.selectbox("Select nutrient for y-axis", selected_nutrients)
    fig_scatter = px.scatter(
        filtered_df,
        x='calories',
        y=scatter_choice,
        hover_name='name',
        labels={'calories': 'Calories', scatter_choice: scatter_choice.title()}
    )
    st.plotly_chart(fig_scatter)

# Section 4: Radar Chart for a Selected Food
st.subheader("🧭 Nutrient Radar for a Selected Food")
food_option = st.selectbox("Choose a food to visualize", filtered_df['name'].dropna().unique())
food_row = filtered_df[filtered_df['name'] == food_option][selected_nutrients].iloc[0]
fig_radar = go.Figure(data=go.Scatterpolar(
    r=food_row.values,
    theta=selected_nutrients,
    fill='toself',
    name=food_option
))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
st.plotly_chart(fig_radar)

# Section 5: Heatmap of Selected Nutrients
st.subheader("🌡️ Nutrient Heatmap")
if len(selected_nutrients) >= 2:
    fig_heat = px.imshow(filtered_df[selected_nutrients].corr(), text_auto=True, aspect="auto")
    st.plotly_chart(fig_heat)

# Section 6: Health Score Leaderboard
st.subheader("💡 Top 10 Healthiest Foods (by custom score)")
top_health = filtered_df[['name', 'health_score']].sort_values(by='health_score', ascending=False).head(10)
fig_health = px.bar(top_health, x='name', y='health_score', labels={'health_score': 'Health Score'})
st.plotly_chart(fig_health)

# Footer
st.markdown("---")
st.markdown("Data Source: [Kaggle - Nutritional Values for Common Foods](https://www.kaggle.com/datasets/trolukovich/nutritional-values-for-common-foods-and-products)")
