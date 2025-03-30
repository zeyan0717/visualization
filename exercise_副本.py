# Interactive nutrition dashboard using Streamlit
import pandas as pd
import streamlit as st
import plotly.express as px

# Load and clean dataset
df = pd.read_csv("/Users/yanze/Desktop/nutrition.csv")

# Convert numeric fields from strings (e.g. '12.3 g') to floats
def extract_number(x):
    try:
        return float(str(x).split()[0])
    except:
        return None

columns_to_clean = [
    'calories', 'protein', 'carbohydrate', 'sugars', 'total_fat', 'sodium',
    'saturated_fat', 'fiber'
]

for col in columns_to_clean:
    if col in df.columns:
        df[col] = df[col].apply(extract_number)

# Sidebar filters
st.sidebar.header("Filter Options")

cal_min, cal_max = int(df['calories'].min()), int(df['calories'].max())
calories_range = st.sidebar.slider("Calories Range", cal_min, cal_max, (cal_min, cal_max))

selected_nutrients = st.sidebar.multiselect(
    "Select nutrients to compare",
    options=['protein', 'carbohydrate', 'sugars', 'total_fat', 'sodium', 'fiber'],
    default=['protein', 'sugars']
)

# Filter data
filtered_df = df[(df['calories'] >= calories_range[0]) & (df['calories'] <= calories_range[1])]

# Main dashboard
st.title("Interactive Nutrition Dashboard")

st.markdown("""
Explore and compare nutritional values of over 8,000 foods. Use the filters to analyze items by calorie content, or compare across selected nutrients.
""")

# Bar chart of selected nutrients
if selected_nutrients:
    st.subheader("Nutrient Comparison")
    mean_values = filtered_df[selected_nutrients].mean().sort_values(ascending=False)
    fig = px.bar(
        x=mean_values.index,
        y=mean_values.values,
        labels={'x': 'Nutrient', 'y': 'Average Content (per 100g)'}
    )
    st.plotly_chart(fig)

# Top 10 foods by protein
st.subheader("Top 10 High-Protein Foods")
top_protein = filtered_df[['name', 'protein']].dropna().sort_values(by='protein', ascending=False).head(10)
fig2 = px.bar(top_protein, x='name', y='protein', labels={'protein': 'Protein (g)'})
st.plotly_chart(fig2)

# Nutrient scatter plot
st.subheader("Scatter Plot: Calories vs Selected Nutrient")
scatter_nutrient = st.selectbox("Select nutrient for y-axis", options=selected_nutrients)
fig3 = px.scatter(filtered_df, x='calories', y=scatter_nutrient, hover_name='name', labels={'calories': 'Calories'})
st.plotly_chart(fig3)

st.markdown("---")
st.markdown("Data Source: [Kaggle - Nutritional Values for Common Foods](https://www.kaggle.com/datasets/trolukovich/nutritional-values-for-common-foods-and-products)")
