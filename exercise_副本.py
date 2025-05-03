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
    # ... (data cleaning steps) ...
    return df_raw

# Load data
df = load_and_clean_data("nutrition.csv")

# --- Sidebar & Navigation ---
st.sidebar.title("Navigation & Settings")
page = st.sidebar.radio(
    "Select Page",
    [
        "Nutrition Analysis", "Nutrition Radar", "Heatmap", "Health Ranking",
        "Food Clustering", "Meal Planner", "Nutrition Network", "Report & Export"
    ],
    help="Select a page to access different features of the nutrition tool."
)

# Filters for most pages
st.sidebar.header("Filter Options")
cal_range = st.sidebar.slider(
    "Calorie Range",
    0,
    int(df['calories'].max()),
    (0, 500),
    help="Filter foods by calorie range."
)
protein_range = st.sidebar.slider(
    "Protein Range (g)",
    0.0,
    float(df['protein'].max()),
    (0.0, 50.0),
    help="Filter foods by protein content."
)
sugar_range = st.sidebar.slider(
    "Sugar Range (g)",
    0.0,
    float(df['sugars'].max()),
    (0.0, 10.0),
    help="Filter foods by sugar content."
)

filtered_df = df[
    (df['calories'] >= cal_range[0]) & (df['calories'] <= cal_range[1]) &
    (df['protein'] >= protein_range[0]) & (df['protein'] <= protein_range[1]) &
    (df['sugars'] >= sugar_range[0]) & (df['sugars'] <= sugar_range[1])
]

default_nutrients = ['protein','carbohydrate','sugars','fat','fiber','water']
selected_nutrients = st.sidebar.multiselect(
    "Select Nutrients",
    default_nutrients,
    default=default_nutrients,
    help="Choose which nutrients to include in analysis and visualizations."
)

# --- Page: Nutrition Analysis ---
if page == "Nutrition Analysis":
    st.title("🥗 Nutrition Analysis")
    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Scatter Plot", "Box Plot"])

    with tab1:
        st.subheader("Average Nutrients")
        if selected_nutrients:
            avg = filtered_df[selected_nutrients].mean().sort_values(ascending=False)
            avg_df = pd.DataFrame({"Nutrient": avg.index, "Average": avg.values})
            fig = px.bar(
                avg_df,
                x="Nutrient",
                y="Average",
                hover_data={"Average":":.2f"},
                labels={"Nutrient":"Nutrient", "Average":"Average Value"}
            )
            st.plotly_chart(fig)

    with tab2:
        st.subheader("Calories vs Nutrient")
        nut = st.selectbox(
            "Pick a nutrient",
            selected_nutrients,
            help="Select nutrient for the scatter plot."
        )
        fig = px.scatter(
            filtered_df,
            x="calories",
            y=nut,
            hover_data=["name","health_score"],
            labels={"calories":"Calories", nut:nut}
        )
        st.plotly_chart(fig)

    with tab3:
        st.subheader("Nutrient Distribution")
        if selected_nutrients:
            melted = filtered_df.melt(
                id_vars=['name'],
                value_vars=selected_nutrients,
                var_name='Nutrient',
                value_name='Value'
            )
            fig = px.box(
                melted,
                x='Nutrient',
                y='Value',
                hover_data=['name'],
                labels={"Nutrient":"Nutrient","Value":"Value"}
            )
            st.plotly_chart(fig)

# --- Page: Nutrition Radar ---
elif page == "Nutrition Radar":
    st.title("🧭 Nutrition Radar")
    food = st.selectbox(
        "Pick Food",
        filtered_df['name'].unique(),
        help="Select a food to view its nutrient profile."
    )
    if food and selected_nutrients:
        row = filtered_df[filtered_df['name']==food][selected_nutrients].iloc[0]
        fig = go.Figure(
            go.Scatterpolar(
                r=row.values,
                theta=selected_nutrients,
                fill='toself',
                name=food
            )
        )
        angle = 180 / len(selected_nutrients)
        fig.update_layout(
            polar=dict(angularaxis=dict(tickangle=angle)),
            title_text=f"Nutrition Radar for {food}"
        )
        st.plotly_chart(fig)
    else:
        st.warning("Select a food and nutrients.")

# --- Page: Heatmap ---
elif page == "Heatmap":
    st.title("🌡️ Nutrition Heatmap")
    if selected_nutrients:
        fig = px.imshow(
            filtered_df[selected_nutrients].corr(),
            text_auto=True,
            labels={"x":"Nutrient","y":"Nutrient","color":"Corr"}
        )
        st.plotly_chart(fig)
    else:
        st.warning("Select nutrients for heatmap.")

# --- Page: Health Ranking ---
elif page == "Health Ranking":
    st.title("⭐ Health Ranking")
    st.write("Feature under construction...")

# --- Page: Food Clustering ---
elif page == "Food Clustering":
    st.title("🍽️ Food Clustering & Dimensionality Reduction")
    if selected_nutrients:
        alg = st.selectbox(
            "Clustering Algorithm",
            ["KMeans","Agglomerative"],
            help="Select the clustering algorithm."
        )
        k_max = st.slider(
            "Max Clusters for Animation",
            2,10,4,
            help="Set maximum number of clusters to animate through."
        )
        animate = st.checkbox(
            "Enable Animation Across Cluster Counts",
            help="Toggle to animate cluster count changes."
        )
        data = filtered_df[selected_nutrients].dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        dr_method = st.radio(
            "Dimensionality Reduction Method",
            [
                "PCA (Principal Component Analysis)",
                "t-SNE (t-Distributed Stochastic Neighbor Embedding)"
            ],
            help="Choose PCA for linear projection or t-SNE for non-linear embedding."
        )
        dr = PCA(n_components=2) if dr_method.startswith("PCA") else TSNE(n_components=2, random_state=0)
        frames = []
        ks = list(range(2,k_max+1)) if animate else [st.slider(
            "Number of Clusters",2,10,4, help="Select number of clusters."
        )]
        for k in ks:
            model = KMeans(n_clusters=k, random_state=0) if alg=="KMeans" else AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(scaled)
            coords = dr.fit_transform(scaled)
            dfp = pd.DataFrame(coords,columns=['Dim1','Dim2'])
            dfp['Cluster']=labels.astype(str)
            dfp['Name']=filtered_df['name'].tolist()
            dfp['Cluster_Count']=k
            frames.append(dfp)
        plot_df = pd.concat(frames)
        if animate:
            fig = px.scatter(
                plot_df,
                x='Dim1',y='Dim2',
                animation_frame='Cluster_Count',
                color='Cluster',
                hover_data=['Name'],
                labels={'Dim1':'Dimension 1','Dim2':'Dimension 2'}
            )
        else:
            fig = px.scatter(
                plot_df,
                x='Dim1',y='Dim2',
                color='Cluster',
                hover_data=['Name'],
                labels={'Dim1':'Dimension 1','Dim2':'Dimension 2'}
            )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig)
        score = silhouette_score(scaled,frames[-1]['Cluster'].astype(int))
        st.write(f"Silhouette Score (k={frames[-1]['Cluster_Count']}): {score:.2f}")
    else:
        st.warning("Select nutrients for clustering.")

# --- Page: Meal Planner ---
elif page == "Meal Planner":
    st.title("📋 Smart Meal Planner")
    fine_tune = st.checkbox(
        "Enable fine-tune mode for portion inputs (0.1 units)",
        value=False,
        help="Toggle to allow finer control over portion sizes."
    )
    step_size = 0.1 if fine_tune else 1.0
    foods = st.multiselect(
        "Pick foods",
        filtered_df['name'].unique(),
        default=filtered_df['name'].sample(3).tolist(),
        help="Select foods to include in your meal plan."
    )
    portions = {
        food: st.number_input(
            f"{food} (100g units)",
            0.0,10.0,1.0,
            step=step_size,
            format="%.1f",
            help="Adjust portion size of each selected food."
        )
        for food in foods
    }
    totals = {
        nut: sum(filtered_df.loc[filtered_df['name']==f,nut].iloc[0]*p for f,p in portions.items())
        for nut in ['protein','carbohydrate','fat']
    }
    st.subheader("Totals (per meal)")
    st.write(totals)
    tp = st.number_input(
        "Protein target (g)",
        0.0,200.0,50.0,
        step=0.1,
        format="%.1f",
        help="Set your protein target for the meal."
    )
    tc = st.number_input(
        "Carbs target (g)",
        0.0,400.0,200.0,
        step=0.1,
        format="%.1f",
        help="Set your carbohydrates target for the meal."
    )
    tf = st.number_input(
        "Fat target (g)",
        0.0,100.0,30.0,
        step=0.1,
        format="%.1f",
        help="Set your fat target for the meal."
    )
    for nut,val,target in zip(['Protein','Carbs','Fat'],totals.values(),[tp,tc,tf]):
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=val,
            gauge={'axis':{'range':[0,target]}},
            title={'text':nut}
        ))
        st.plotly_chart(fig)
    if st.button("Optimize Meal for Targets") and foods:
        with st.spinner("Optimizing meal for your targets..."):
            idx = filtered_df[filtered_df['name'].isin(foods)].index
            A = [[-filtered_df.loc[i,n] for i in idx] for n in ['protein','carbohydrate','fat']]
            b = [-tp,-tc,-tf]
            res = linprog(c=[1]*len(idx),A_ub=A,b_ub=b,bounds=[(0,None)]*len(idx))
        if res.success:
            sol = dict(zip(filtered_df.loc[idx,'name'],res.x))
            st.success("Optimization Complete!")
            st.write(sol)
        else:
            st.error("Optimization failed. Please adjust your targets or food selection.")

# --- Page: Nutrition Network ---
elif page == "Nutrition Network":
    st.title("🔗 Dynamic Nutrition Network")
    if len(selected_nutrients)>=2:
        corr = filtered_df[selected_nutrients].corr().abs()
        thresh = st.slider(
            "Min correlation (recommended ≥ 0.3)",
            0.0,1.0,0.3,
            help="Set the minimum absolute correlation threshold for edges."
        )
        G = nx.Graph()
        for n in selected_nutrients:
            G.add_node(n)
        for i,a in enumerate(selected_nutrients):
            for j,b in enumerate(selected_nutrients[i+1:],start=i+1):
                if corr.iloc[i,j]>=thresh:
                    G.add_edge(a,b,weight=corr.iloc[i,j])
        pos = nx.spring_layout(G)
        edge_x,edge_y=[],[]
        for u,v in G.edges():
            x0,y0 = pos[u]; x1,y1 = pos[v]
            edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
        node_x=[pos[n][0] for n in G.nodes()]; node_y=[pos[n][1] for n in G.nodes()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode='lines',hoverinfo='none'))
        fig.add_trace(go.Scatter(x=node_x,y=node_y,mode='markers+text',
                                 text=list(G.nodes()),textposition='top center',
                                 marker={'size':20},
                                 hoverinfo='text',hovertext=[f"Nutrient: {n}" for n in G.nodes()]))
        fig.update_layout(showlegend=False,title_text="Nutrition Correlation Network")
        st.plotly_chart(fig)
    else:
        st.warning("Select at least two nutrients.")

# --- Page: Report & Export ---
elif page == "Report & Export":
    st.title("📤 Report Export & Sharing")
    if st.button("Generate PDF Report"):
        buffer = io.BytesIO()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial","B",16)
        pdf.cell(0,10,"Nutrition Analysis Report",ln=True,align="C")
        fig = px.imshow(
            filtered_df[selected_nutrients].corr(),
            text_auto=True,
            labels={"x":"Nutrient","y":"Nutrient","color":"Correlation"}
        )
        tmp = tempfile.NamedTemporaryFile(suffix='.png',delete=False)
        fig.write_image(tmp.name)
        pdf.image(tmp.name, w=180)
        pdf.output(buffer)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
    if st.button("Download Filtered Data CSV"):
        csv = filtered_df[selected_nutrients].to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("Data source: Kaggle - Nutritional values for common foods and products")
