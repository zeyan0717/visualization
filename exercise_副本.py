# Interactive nutrition dashboard using Streamlit
import pandas as pd
import streamlit as st
import plotly.express as px

# Load and clean dataset
df_raw = pd.read_csv("nutrition.csv")
df = df_raw.copy()

# Define columns to clean
columns_to_convert = [
    'calories', 'protein', 'carbohydrate', 'sugars', 'total_fat', 'sodium',
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

# Section 2: Top 10 Protein-rich Foods
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

# Footer
st.markdown("---")
st.markdown("Data Source: [Kaggle - Nutritional Values for Common Foods](https://www.kaggle.com/datasets/trolukovich/nutritional-values-for-common-foods-and-products)")
