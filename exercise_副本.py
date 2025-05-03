import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import networkx as nx
from scipy.optimize import linprog
import io
import base64
from fpdf import FPDF
import tempfile

# --- Data loading and cleaning ---
@st.cache_data
def load_and_clean_data(path: str):
    df_raw = pd.read_csv(path)
    # Convert key columns to numeric, coerce errors into NaN
    numeric_cols = ['calories', 'protein', 'carbohydrate', 'sugars', 'fat', 'fiber', 'water', 'health_score']
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    # Replace all NaN in numeric columns with 0
    df_raw[numeric_cols] = df_raw[numeric_cols].fillna(0)
    return df_raw

# Load data
df = load_and_clean_data("nutrition.csv")

# --- Sidebar & Navigation ---
st.sidebar.title("Navigation & Settings")
page = st.sidebar.radio(
    "Select Page",
    [
        "Nutrition Analysis",
        "Nutrition Radar",
        "Heatmap",
        "Health Ranking",
        "Food Clustering",
        "Meal Planner",
        "Nutrition Network",
        "Report & Export"
    ],
    help="Select a page to access different features of the nutrition tool."
)

# --- Filter Options ---
st.sidebar.header("Filter Options")
# Determine slider bounds safely
min_cal = int(df['calories'].min()) if not df['calories'].empty else 0
max_cal = int(df['calories'].max()) if not df['calories'].empty else 0
cal_range = st.sidebar.slider(
    "Calorie Range",
    0,
    max_cal,
    (0, min(500, max_cal)),
    help="Filter foods by calorie range."
)
min_pro = float(df['protein'].min()) if not df['protein'].empty else 0.0
max_pro = float(df['protein'].max()) if not df['protein'].empty else 0.0
protein_range = st.sidebar.slider(
    "Protein Range (g)",
    0.0,
    max_pro,
    (0.0, min(50.0, max_pro)),
    help="Filter foods by protein content."
)
min_sug = float(df['sugars'].min()) if not df['sugars'].empty else 0.0
max_sug = float(df['sugars'].max()) if not df['sugars'].empty else 0.0
sugar_range = st.sidebar.slider(
    "Sugar Range (g)",
    0.0,
    max_sug,
    (0.0, min(10.0, max_sug)),
    help="Filter foods by sugar content."
)

# Apply filters
df_filtered = df[
    (df['calories'] >= cal_range[0]) & (df['calories'] <= cal_range[1]) &
    (df['protein'] >= protein_range[0]) & (df['protein'] <= protein_range[1]) &
    (df['sugars'] >= sugar_range[0]) & (df['sugars'] <= sugar_range[1])
].reset_index(drop=True)

# Nutrient selection
default_nutrients = ['protein', 'carbohydrate', 'sugars', 'fat', 'fiber', 'water']
selected_nutrients = st.sidebar.multiselect(
    "Select Nutrients",
    default_nutrients,
    default=default_nutrients,
    help="Choose which nutrients to include in analysis and visualizations."
)

# --- Page implementations ---
# 1. Nutrition Analysis
if page == "Nutrition Analysis":
    st.title("🥗 Nutrition Analysis")
    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Scatter Plot", "Box Plot"])

    with tab1:
        st.subheader("Average Nutrients")
        if selected_nutrients and not df_filtered.empty:
            avg = df_filtered[selected_nutrients].mean().sort_values(ascending=False)
            avg_df = pd.DataFrame({"Nutrient": avg.index, "Average": avg.values})
            fig = px.bar(avg_df, x="Nutrient", y="Average",
                         hover_data={"Average": ":.2f"},
                         labels={"Nutrient": "Nutrient", "Average": "Average Value"})
            fig.update_layout(xaxis_title="Nutrient", yaxis_title="Average Value")
            st.plotly_chart(fig)
        else:
            st.warning("No data available for selected filters/nutrients.")

    with tab2:
        st.subheader("Calories vs Nutrient")
        if selected_nutrients and not df_filtered.empty:
            nut = st.selectbox("Pick a nutrient", selected_nutrients,
                               help="Select nutrient for the scatter plot.")
            fig = px.scatter(df_filtered, x="calories", y=nut,
                             hover_data=["name", "health_score"],
                             labels={"calories": "Calories", nut: nut})
            fig.update_layout(xaxis_title="Calories", yaxis_title=nut)
            st.plotly_chart(fig)
        else:
            st.warning("Select nutrients and ensure data is available.")

    with tab3:
        st.subheader("Nutrient Distribution")
        if selected_nutrients and not df_filtered.empty:
            melted = df_filtered.melt(id_vars=['name'], value_vars=selected_nutrients,
                                      var_name='Nutrient', value_name='Value')
            fig = px.box(melted, x='Nutrient', y='Value', hover_data=['name'],
                         labels={"Nutrient": "Nutrient", "Value": "Value"})
            fig.update_layout(xaxis_title="Nutrient", yaxis_title="Value")
            st.plotly_chart(fig)
        else:
            st.warning("Select nutrients and ensure data is available.")

# 2. Nutrition Radar
elif page == "Nutrition Radar":
    st.title("🧭 Nutrition Radar")
    if selected_nutrients and not df_filtered.empty:
        food = st.selectbox("Pick Food", df_filtered['name'].unique(),
                            help="Select a food to view its nutrient profile.")
        if food:
            row = df_filtered[df_filtered['name'] == food][selected_nutrients].iloc[0]
            fig = go.Figure(go.Scatterpolar(r=row.values, theta=selected_nutrients,
                                            fill='toself', name=food))
            angle = 180 / len(selected_nutrients)
            fig.update_layout(polar=dict(angularaxis=dict(tickangle=angle)),
                              title_text=f"Nutrition Radar for {food}")
            st.plotly_chart(fig)
        else:
            st.warning("Please select a food.")
    else:
        st.warning("Select nutrients and ensure data is available.")

# 3. Heatmap
elif page == "Heatmap":
    st.title("🌡️ Nutrition Heatmap")
    if selected_nutrients and not df_filtered.empty:
        corr = df_filtered[selected_nutrients].corr()
        fig = px.imshow(corr, text_auto=True,
                        labels={"x": "Nutrient", "y": "Nutrient", "color": "Corr"})
        st.plotly_chart(fig)
    else:
        st.warning("Select nutrients and ensure data is available.")

# 4. Health Ranking
elif page == "Health Ranking":
    st.title("⭐ Health Ranking")
    st.info("This feature is under construction.")

# 5. Food Clustering
elif page == "Food Clustering":
    st.title("🍽️ Food Clustering & Dimensionality Reduction")
    if selected_nutrients and not df_filtered.empty:
        alg = st.selectbox("Clustering Algorithm", ["KMeans", "Agglomerative"],
                           help="Select the clustering algorithm.")
        k_max = st.slider("Max Clusters for Animation", 2, 10, 4,
                          help="Maximum number of clusters to animate.")
        animate = st.checkbox("Enable Animation Across Cluster Counts",
                              help="Toggle animation for cluster counts.")
        vals = df_filtered[selected_nutrients]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(vals)
        dr_method = st.radio("Dimensionality Reduction Method",
                              ["PCA (Principal Component Analysis)",
                               "t-SNE (t-Distributed Stochastic Neighbor Embedding)"],
                              help="Choose linear or non-linear embedding.")
        dr = PCA(n_components=2) if dr_method.startswith("PCA") else TSNE(n_components=2, random_state=0)
        frames = []
        ks = list(range(2, k_max + 1)) if animate else [st.slider("Number of Clusters", 2, 10, 4)]
        for k in ks:
            model = KMeans(n_clusters=k, random_state=0) if alg == "KMeans" else AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(scaled)
            coords = dr.fit_transform(scaled)
            dfp = pd.DataFrame(coords, columns=['Dim1', 'Dim2'])
            dfp['Cluster'] = labels.astype(str)
            dfp['Name'] = df_filtered['name'].tolist()
            dfp['Cluster_Count'] = k
            frames.append(dfp)
        plot_df = pd.concat(frames)
        if animate:
            fig = px.scatter(plot_df, x='Dim1', y='Dim2', animation_frame='Cluster_Count',
                             color='Cluster', hover_data=['Name'],
                             labels={'Dim1': 'Dimension 1', 'Dim2': 'Dimension 2'})
        else:
            fig = px.scatter(plot_df, x='Dim1', y='Dim2', color='Cluster', hover_data=['Name'],
                             labels={'Dim1': 'Dimension 1', 'Dim2': 'Dimension 2'})
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig)
        score = silhouette_score(scaled, frames[-1]['Cluster'].astype(int))
        st.write(f"Silhouette Score (k={frames[-1]['Cluster_Count']}): {score:.2f}")
    else:
        st.warning("Select nutrients and ensure data is available.")

# 6. Meal Planner
elif page == "Meal Planner":
    st.title("📋 Smart Meal Planner")
    fine_tune = st.checkbox("Enable fine-tune mode for portion inputs (0.1 units)", value=False,
                            help="Toggle fine control over portion sizes.")
    step = 0.1 if fine_tune else 1.0
    foods = st.multiselect("Pick foods", df_filtered['name'].unique(),
                            default=df_filtered['name'].sample(3).tolist(),
                            help="Select foods for your meal plan.")
    portions = {f: st.number_input(f"{f} (100g)", min_value=0.0, max_value=10.0, value=1.0,
                                    step=step, format="%.1f",
                                    help="Portion size in 100g units.") for f in foods}
    totals = {nut: sum(df_filtered.loc[df_filtered['name']==f, nut].iloc[0] * p for f, p in portions.items())
              for nut in ['protein', 'carbohydrate', 'fat']}
    st.subheader("Totals (per meal)")
    st.write(totals)
    tp = st.number_input("Protein target (g)", min_value=0.0, max_value=200.0, value=50.0,
                          step=0.1, format="%.1f", help="Protein target.")
    tc = st.number_input("Carbs target (g)", min_value=0.0, max_value=400.0, value=200.0,
                          step=0.1, format="%.1f", help="Carbs target.")
    tf = st.number_input("Fat target (g)", min_value=0.0, max_value=100.0, value=30.0,
                          step=0.1, format="%.1f", help="Fat target.")
    for nut, val, targ in zip(['Protein','Carbs','Fat'], totals.values(), [tp, tc, tf]):
        fig = go.Figure(go.Indicator(mode='gauge+number', value=val,
                                     gauge={'axis': {'range': [0, targ]}},
                                     title={'text': nut}))
        st.plotly_chart(fig)
    if st.button("Optimize Meal for Targets") and foods:
        with st.spinner("Optimizing..."):
            idx = df_filtered[df_filtered['name'].isin(foods)].index
            A = [[-df_filtered.loc[i, n] for i in idx] for n in ['protein','carbohydrate','fat']]
            b = [-tp, -tc, -tf]
            res = linprog(c=[1]*len(idx), A_ub=A, b_ub=b, bounds=[(0,None)]*len(idx))
        if res.success:
            sol = dict(zip(df_filtered.loc[idx,'name'], res.x))
            st.success("Optimization Complete!")
            st.write(sol)
        else:
            st.error("Optimization failed. Adjust targets or selection.")

# 7. Nutrition Network
elif page == "Nutrition Network":
    st.title("🔗 Nutrition Correlation Network")
    if selected_nutrients and len(selected_nutrients) >= 2 and not df_filtered.empty:
        corr = df_filtered[selected_nutrients].corr().abs()
        thresh = st.slider("Min correlation (≥0.3)", 0.0, 1.0, 0.3,
                           help="Filter edges by absolute correlation.")
        G = nx.Graph()
        for n in selected_nutrients:
            G.add_node(n)
        for i, a in enumerate(selected_nutrients):
            for j, b in enumerate(selected_nutrients[i+1:], start=i+1):
                if corr.iloc[i,j] >= thresh:
                    G.add_edge(a, b, weight=corr.iloc[i,j])
        pos    = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for u,v in G.edges():
            x0,y0 = pos[u]; x1,y1 = pos[v]
            edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', hoverinfo='none'))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                                 textposition='top center', marker={'size

