import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Data loading and cleaning function
@st.cache_data
def load_and_clean_data():
    df_raw = pd.read_csv("nutrition.csv")
    df = df_raw.copy()

    # Columns to convert to numeric
    columns_to_convert = [
        'calories', 'protein', 'carbohydrate', 'sugars', 'sodium',
        'fiber', 'fat', 'water'
    ]

    # Convert columns to numeric
    def extract_numeric(val):
        try:
            return float(str(val).split()[0].replace(',', ''))
        except:
            return None

    for col in columns_to_convert:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)

    # Drop rows with missing core nutrition fields
    df.dropna(subset=['calories', 'protein', 'carbohydrate', 'sugars'], inplace=True)

    # Calculate health score
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

# Load data
df = load_and_clean_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Nutrition Analysis", "Nutrition Radar", "Heatmap", "Health Ranking", "AI Recommendation"]
)

# --- Sidebar filters ---
st.sidebar.header("Filter Options")

# 1) Calorie range
cal_range = st.sidebar.slider(
    "Calorie Range",
    0,
    int(df['calories'].max()),
    (0, 500)
)

# 2) Protein range
protein_range = st.sidebar.slider(
    "Protein Range (g per 100g)",
    0.0,
    float(df["protein"].max()),
    (0.0, 50.0)
)

# 3) Sugar range
sugar_range = st.sidebar.slider(
    "Sugar Range (g per 100g)",
    0.0,
    float(df["sugars"].max()),
    (0.0, 10.0)
)

# Apply filters
filtered_df = df[
    (df['calories'] >= cal_range[0]) & (df['calories'] <= cal_range[1]) &
    (df['protein'] >= protein_range[0]) & (df['protein'] <= protein_range[1]) &
    (df['sugars'] >= sugar_range[0]) & (df['sugars'] <= sugar_range[1])
]

# Shared multiselect for nutrients
default_nutrients = ['protein', 'carbohydrate', 'sugars', 'fat', 'fiber', 'water', 'sodium']
st.sidebar.write("---")
selected_nutrients = st.sidebar.multiselect(
    "Select Nutrients for Visuals",
    default_nutrients,
    default=['protein', 'sugars']
)

# --- Page contents ---
if page == "Nutrition Analysis":
    st.title("🥗 Nutrition Analysis")

    # Use tabs to provide multiple chart types
    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Scatter Plot", "Box Plot"])

    with tab1:
        st.subheader("Average Nutrients (Bar Chart)")
        if selected_nutrients:
            # Calculate average for selected nutrients
            avg_vals = filtered_df[selected_nutrients].mean().sort_values(ascending=False)

            # Turn the Series into a DataFrame for better hover info
            avg_df = pd.DataFrame({
                "Nutrient": avg_vals.index,
                "Average": avg_vals.values
            })

            fig_bar = px.bar(
                avg_df,
                x="Nutrient",
                y="Average",
                labels={"Nutrient": "Nutrient", "Average": "Average Content per 100g"},
                hover_data={"Nutrient": True, "Average": ":.2f"}
            )
            st.plotly_chart(fig_bar)

    with tab2:
        st.subheader("Calorie vs Selected Nutrient (Scatter Plot)")
        # Let the user pick one nutrient to compare with calories
        nutrient_for_scatter = st.selectbox("Select a nutrient to compare with calories", selected_nutrients)

        fig_scatter = px.scatter(
            filtered_df,
            x="calories",
            y=nutrient_for_scatter,
            hover_data=["name", "health_score"],
            labels={"calories": "Calories", nutrient_for_scatter: nutrient_for_scatter}
        )
        st.plotly_chart(fig_scatter)

    with tab3:
        st.subheader("Distribution of Nutrients (Box Plot)")
        if selected_nutrients:
            # Box plot for multiple nutrients
            # We might want to melt the dataframe to plot them in a single box plot
            melted = filtered_df.melt(
                id_vars=['name'],
                value_vars=selected_nutrients,
                var_name="Nutrient",
                value_name="Value"
            )
            fig_box = px.box(
                melted,
                x="Nutrient",
                y="Value",
                labels={"Nutrient": "Nutrient", "Value": "Value (per 100g)"}
            )
            st.plotly_chart(fig_box)

elif page == "Nutrition Radar":
    st.title("🧭 Nutrition Radar")
    food_option = st.selectbox("Select Food", filtered_df['name'].dropna().unique())
    if food_option:
        if selected_nutrients:
            food_row = filtered_df[filtered_df['name'] == food_option][selected_nutrients].iloc[0]
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=food_row.values,
                theta=selected_nutrients,
                fill='toself',
                name=food_option,
                hoverinfo='all'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=False
            )
            st.plotly_chart(fig_radar)
        else:
            st.write("Please select at least one nutrient in the sidebar.")

elif page == "Heatmap":
    st.title("🌡️ Nutrient Correlation Heatmap")
    if len(selected_nutrients) >= 2:
        corr_df = filtered_df[selected_nutrients].corr()
        fig_heat = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation between selected nutrients"
        )
        st.plotly_chart(fig_heat)
    else:
        st.write("Please select at least two nutrients to see the correlation heatmap.")

elif page == "Health Ranking":
    st.title("🏆 Health Score Ranking")
    # Show top 10 by health score
    top_health = filtered_df[['name', 'health_score']]\
        .sort_values(by='health_score', ascending=False).head(10)
    fig_health = px.bar(
        top_health,
        x='name',
        y='health_score',
        labels={'name': 'Food', 'health_score': 'Health Score'},
        hover_data={'name': True, 'health_score': ':.2f'}
    )
    st.plotly_chart(fig_health)

    # Also consider showing data in a table
    st.dataframe(top_health)

elif page == "AI Recommendation":
    st.title("🤖 AI Nutrition Recommendation")

    # Additional user info could be gathered here (e.g., weight, goals, etc.)
    goal = st.radio(
        "Select Your Goal:",
        ["Build Muscle", "Lose Fat", "Control Sugar", "High-Fiber Diet"]
    )

    if goal == "Build Muscle":
        recs = filtered_df.sort_values(by='protein', ascending=False).head(10)
    elif goal == "Lose Fat":
        recs = filtered_df[
            (filtered_df['fat'] < 5) & (filtered_df['sugars'] < 5)
        ].sort_values(by='health_score', ascending=False).head(10)
    elif goal == "Control Sugar":
        recs = filtered_df[
            filtered_df['sugars'] < 2
        ].sort_values(by='health_score', ascending=False).head(10)
    else:  # "High-Fiber Diet"
        recs = filtered_df[
            filtered_df['fiber'] > 5
        ].sort_values(by='health_score', ascending=False).head(10)

    st.markdown("**Recommended Foods:**")
    # Let user choose columns to display
    cols_to_show = st.multiselect(
        "Columns to show in recommendation table",
        df.columns.tolist(),
        default=["name", "protein", "fiber", "fat", "sugars", "calories"]
    )
    st.dataframe(recs[cols_to_show])

    # Optional: visualize recommended foods with a bar chart comparing some nutrient
    recommended_nutrient = st.selectbox(
        "Compare recommended foods by nutrient",
        ["protein", "calories", "fat", "sugars", "fiber"]
    )
    rec_fig = px.bar(
        recs,
        x="name",
        y=recommended_nutrient,
        hover_data=["health_score", "calories", "protein", "fat", "sugars", "fiber"],
        labels={"name": "Food", recommended_nutrient: recommended_nutrient.capitalize()},
        title=f"Top 10 Foods for {goal} (by {recommended_nutrient})"
    )
    st.plotly_chart(rec_fig)

# Footer
st.markdown("---")
st.markdown(
    "Data source: [Kaggle - Nutritional values for common foods and products]"
    "(https://www.kaggle.com/datasets/trolukovich/nutritional-values-for-common-foods-and-products)"
)
