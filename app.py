import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set Seaborn/Matplotlib to white background to match Colab
sns.set_theme(style="whitegrid")
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Remove custom CSS that forces chart backgrounds
# (Comment out or remove the MAKE PLOT BACKGROUNDS WHITE block in local_css)
def local_css():
    st.markdown(
        """
        <style>
        /* -------- BACKGROUND -------- */
        .stApp {
            background: linear-gradient(to bottom, #001f3f, #003366, #004080);
            background-attachment: fixed;
            background-size: cover;
            font-family: 'Segoe UI', sans-serif;
            color: white;
        }
        /* -------- MAIN CONTAINER CARD -------- */
        .block-container {
            background-color: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            padding: 0;
            border-radius: 0;
            box-shadow: none;
            color: #f1f1f1;
            width: 100vw !important;
            min-height: 100vh !important;
            max-width: 100vw !important;
        }
        /* -------- HEADINGS -------- */
        h1, h2, h3, h4 {
            color: #ffffff;
            font-family: 'Segoe UI Semibold', sans-serif;
            letter-spacing: 0.5px;
        }
        /* -------- TABS -------- */
        [data-baseweb="tab"] {
            font-size: 16px;
            font-weight: 500;
            color: #e6f0ff;
        }
        [data-baseweb="tab"]:hover {
            color: #66ccff;
        }
        [data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
            font-weight: bold;
            color: #66ccff;
        }
        /* -------- CHART BACKGROUND -------- */
        .element-container:has(.js-plotly-plot) {
            border-radius: 0;
            padding: 0;
            background-color: rgba(255, 255, 255, 0.03);
            box-shadow: none;
        }
        .js-plotly-plot svg {
            border-radius: 0;
        }
        /* -------- FOOTER/HDR -------- */
        footer, header {
            visibility: hidden;
        }
        /* -------- TAB ANIMATION -------- */
        .css-1cpxqw2, .stTabs {
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        /* -------- MAKE PLOT BACKGROUNDS WHITE -------- */
        /* .element-container:has(.js-plotly-plot), .element-container:has(canvas) {
            background-color: white !important;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        .js-plotly-plot .plotly {
            background-color: white !important;
        } */
        /* Center the tab bar */
        .stTabs [role="tablist"] {
            display: flex;
            justify-content: center;
        }
        /* Slightly center subheaders/titles under each tab */
        .block-container h2 {
            margin-left: 2vw;
        }
        /* Center the main title and subtitle */
        .block-container h1, .block-container h4 {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Page config
st.set_page_config(page_title="Decoding Passenger Satisfaction", layout="wide")
local_css()

# Load data
df = pd.read_csv('your_dataset.csv')
df.columns = df.columns.str.strip()

# Title
st.title("✈️ Decoding Passenger Satisfaction")
st.markdown("#### Analyze What Drives Passenger Experience in Air Travel")
st.markdown("---")

# Tabs
tabs = st.tabs(["Histogram", "Scatter plot", "Violin plot", "Bubble scatter plot", "heatmap", "Sankey diagram", "scatter plot after clustering and dimensionality reduction", "Radar chart"])

# --- Tab 1: Histogram ---
with tabs[0]:
    st.subheader("Visualization 1: Satisfaction by Travel Class")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Satisfaction', hue='Class', multiple='dodge', palette='Set2', kde=True, ax=ax)
    ax.set_title('Passenger Satisfaction by Class')
    ax.set_xlabel('Satisfaction')
    ax.set_ylabel('Count')
    st.pyplot(fig, use_container_width=True)
    st.caption("Business class passengers show higher satisfaction, while Economy class sees more dissatisfaction.")

# --- Tab 2: Delays vs Satisfaction ---
with tabs[1]:
    st.subheader("Visualization 2: Departure Delay vs Satisfaction")
    fig_delay, ax_delay = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        x='Departure Delay',
        y='Satisfaction',
        data=df,
        hue='Class',
        palette='Set2',
        s=100,
        edgecolor='black',
        alpha=0.8,
        ax=ax_delay
    )
    ax_delay.set_title('Impact of Departure Delay on Passenger Satisfaction by Class', fontsize=16)
    ax_delay.set_xlabel('Departure Delay (minutes)', fontsize=14)
    ax_delay.set_ylabel('Satisfaction Score', fontsize=14)
    ax_delay.legend(title='Travel Class')
    ax_delay.grid(True, linestyle='-', alpha=0.5)
    st.pyplot(fig_delay, use_container_width=True)
    st.caption("Even short delays can reduce satisfaction, but major dissatisfaction occurs when delays exceed 100 minutes.")

# --- Tab 3: Arrival Delay vs Satisfaction ---
with tabs[2]:
    st.subheader("Visualization 3: Arrival Delay Impact on Satisfaction")
    df['Satisfaction'] = df['Satisfaction'].str.strip().str.lower()
    df['Satisfaction_Label'] = df['Satisfaction'].map({
        'satisfied': 'Satisfied',
        'neutral or dissatisfied': 'Neutral or Dissatisfied'
    })
    if 'Satisfaction_Label' in df.columns:
        fig_violin = px.violin(
            df,
            y="Arrival Delay",
            x="Satisfaction_Label",
            box=True,
            points="all",
            color="Satisfaction_Label",
            hover_data=["Flight Distance", "Type of Travel", "Class", "Age"],
            color_discrete_map={
                "Satisfied": "#B388EB",
                "Neutral or Dissatisfied": "#61C0BF"
            },
            template='plotly_white'
        )
        fig_violin.update_layout(
            title="Impact of Arrival Delays on Passenger Satisfaction",
            yaxis_title="Arrival Delay (Minutes)",
            xaxis_title="Passenger Satisfaction Level",
            transition=dict(duration=500, easing="cubic-in-out"),
            width=850,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color="black"),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'), showgrid=True, linecolor='black', zerolinecolor='black'),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'), showgrid=True, linecolor='black', zerolinecolor='black'),
            legend=dict(font=dict(color='black'))
        )
        fig_violin.update_traces(meanline_visible=True)
        st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.warning("Satisfaction_Label column not found.")

# --- Tab 4: Satisfied Despite High Delays (Anomalies) ---
with tabs[3]:
    st.subheader("Visualization 4: High Delay but Still Satisfied (Anomalies)")
    df['Departure Delay'] = df['Departure Delay'].fillna(0)
    df['Arrival Delay'] = df['Arrival Delay'].fillna(0)
    df['Total_Delay'] = df['Departure Delay'] + df['Arrival Delay']
    threshold_delay = df['Total_Delay'].quantile(0.90)
    anomalies = df[(df['Total_Delay'] > threshold_delay) & (df['Satisfaction'].str.lower() == 'satisfied')]
    fig_anomaly = px.scatter(
        df,
        x='Total_Delay',
        y='Flight Distance',
        color='Satisfaction',
        color_discrete_map={
            'satisfied': '#FBB4AE',
            'neutral or dissatisfied': '#B3CDE3'
        },
        opacity=0.5,
        size='Age',
        size_max=20,
        template='plotly_white',
        title="High Delay but Still Satisfied (Anomalies)",
        labels={
            "Total_Delay": "Total Delay (mins)",
            "Flight Distance": "Flight Distance"
        }
    )
    fig_anomaly.add_trace(go.Scatter(
        x=anomalies['Total_Delay'],
        y=anomalies['Flight Distance'],
        mode='markers',
        name='Happy Despite Delay',
        marker=dict(
            symbol='star',
            size=16,
            color='gold',
            opacity=0.9,
            line=dict(width=0)
        ),
        hovertext='Anomaly: Satisfied with high delay',
        showlegend=True
    ))
    fig_anomaly.update_layout(
        legend_title="Satisfaction",
        width=950,
        height=650,
        font=dict(family='Arial', size=14, color='black'),
        xaxis=dict(showgrid=True, gridcolor='lightgray', title_font=dict(color='black'), tickfont=dict(color='black'), linecolor='black', zerolinecolor='black'),
        yaxis=dict(showgrid=True, gridcolor='lightgray', title_font=dict(color='black'), tickfont=dict(color='black'), linecolor='black', zerolinecolor='black'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(font=dict(color='black'))
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)
    st.caption("This violin plot shows how arrival delays skew dissatisfaction. Longer delays especially frustrate neutral/dissatisfied travelers.")

# --- Tab 5: Service Feature Correlation (Heatmap) ---
with tabs[4]:
    st.subheader("Visualization 5: Correlation Between Service Features")
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    if "Total_Delay" in numerical_cols:
        numerical_cols.remove("Total_Delay")
    corr_matrix = df[numerical_cols].corr().round(2)
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        zmin=-1,
        zmax=1,
        colorscale='RdBu',
        colorbar=dict(title='Correlation'),
        hovertemplate='Feature 1: %{y}<br>Feature 2: %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    fig_heatmap.update_layout(
        title='Interactive Correlation between Service Features',
        xaxis_title="Service Features",
        yaxis_title="Service Features",
        width=850,
        height=850,
        xaxis=dict(
            tickangle=90,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            tickfont=dict(size=11, color='black'),
            scaleanchor='y',
            title_font=dict(color='black'),
            linecolor='black',
            zerolinecolor='black'
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            tickfont=dict(size=11, color='black'),
            title_font=dict(color='black'),
            linecolor='black',
            zerolinecolor='black'
        ),
        font=dict(family='Segoe UI', color='black'),
        margin=dict(t=50, l=80, r=50, b=80),
        plot_bgcolor='white',
        paper_bgcolor='white',
        transition=dict(duration=500, easing='cubic-in-out'),
        legend=dict(font=dict(color='black'))
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption("This clean heatmap shows correlations among numeric features without grid distractions.")

# --- Tab 6: Travel Type to Satisfaction (Sankey Diagram) ---
with tabs[5]:
    st.subheader("Visualization 6: Passenger Travel Type ➔ Satisfaction Flow")
    sankey_data = df.groupby(['Type of Travel', 'Satisfaction']).size().reset_index(name='count')
    labels = sankey_data['Type of Travel'].unique().tolist() + sankey_data['Satisfaction'].unique().tolist()
    label_map = {label: idx for idx, label in enumerate(labels)}
    source = sankey_data['Type of Travel'].map(label_map)
    target = sankey_data['Satisfaction'].map(label_map)
    value = sankey_data['count']
    pastel_colors = ['#4ABDAC', '#F7B733', '#3B3B98', '#FC4A1A', '#ffb6b9', '#a2d5f2', '#ffe156', '#a0e426', '#f85f73', '#ff9a00']
    node_colors = pastel_colors * (len(labels) // len(pastel_colors) + 1)
    node_colors = node_colors[:len(labels)]
    link_colors = [
        'rgba(173, 216, 230, 0.6)' if sat.strip().lower() == 'satisfied' else 'rgba(255, 213, 128, 0.6)'
        for sat in sankey_data['Satisfaction']
    ]
    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])
    fig_sankey.update_layout(
        title_text="Passenger Travel Type ➔ Satisfaction Flow",
        font=dict(size=14, family='Arial', color='black'),
        margin=dict(l=30, r=30, t=50, b=30),
        width=900,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(font=dict(color='black')),
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'), linecolor='black', zerolinecolor='black'),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'), linecolor='black', zerolinecolor='black')
    )
    st.plotly_chart(fig_sankey, use_container_width=True)
    st.caption("This Sankey diagram shows how different travel types (e.g., Business vs. Personal) influence satisfaction levels. The thicker the flow, the greater the number of passengers in that path.")

# --- Tab 7: Passenger Segments (PCA + KMeans Clustering) ---
with tabs[6]:
    st.subheader("Visualization 9: Passenger Segments - Who They Are")
    service_features = [
        'In-flight Entertainment', 'Seat Comfort', 'Cleanliness',
        'Food and Drink', 'In-flight Wifi Service', 'Leg Room Service'
    ]
    features = service_features + ['Flight Distance', 'Departure Delay', 'Arrival Delay', 'Age']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    df_pca = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters,
        'Class': df.loc[X.index, 'Class'],
        'Type of Travel': df.loc[X.index, 'Type of Travel'],
        'Age': df.loc[X.index, 'Age'],
        'Flight Distance': df.loc[X.index, 'Flight Distance']
    })
    fig_pca = px.scatter(
        df_pca,
        x='PCA1',
        y='PCA2',
        color=df_pca['Cluster'].astype(str),
        hover_data={
            'Class': True,
            'Type of Travel': True,
            'Age': True,
            'Flight Distance': True,
            'PCA1': False,
            'PCA2': False,
            'Cluster': False
        },
        title="Passenger Segments: Who They Are (KMeans + PCA)",
        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
        width=900,
        height=700,
        template='plotly_white'
    )
    fig_pca.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    fig_pca.update_layout(
        legend_title="Passenger Segment",
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, title_font=dict(color='black'), tickfont=dict(color='black'), linecolor='black', zerolinecolor='black'),
        yaxis=dict(showgrid=False, title_font=dict(color='black'), tickfont=dict(color='black'), linecolor='black', zerolinecolor='black'),
        font=dict(color='black'),
        legend=dict(font=dict(color='black'))
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption("This plot segments passengers based on service preferences and travel behavior. Clusters are created using KMeans, and reduced to 2D with PCA.")

# --- Tab 8: Cluster Profiles by Service Preferences (Radar Chart) ---
with tabs[7]:
    st.subheader("Visualization 10: Service Profile by Passenger Segment (Radar Chart)")
    if 'Cluster' not in df.columns:
        st.warning("Please generate clusters in the previous tab before viewing this chart.")
    else:
        service_features = [
            'In-flight Entertainment', 'Seat Comfort', 'Cleanliness',
            'Food and Drink', 'In-flight Wifi Service', 'Leg Room Service'
        ]
        cluster_service_means = df.groupby('Cluster')[service_features].mean().reset_index()
        fig_radar_cluster = go.Figure()
        for cluster in cluster_service_means['Cluster']:
            fig_radar_cluster.add_trace(go.Scatterpolar(
                r=cluster_service_means[cluster_service_means['Cluster'] == cluster][service_features].values.flatten(),
                theta=service_features,
                fill='toself',
                name=f'Cluster {cluster}'
            ))
        fig_radar_cluster.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            title="Service Profile by Passenger Segment (Radar Chart)",
            width=800,
            height=600,
            font=dict(color='black'),
            template='plotly_white',
            legend=dict(font=dict(color='black')),
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'), linecolor='black', zerolinecolor='black'),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'), linecolor='black', zerolinecolor='black')
        )
        st.plotly_chart(fig_radar_cluster, use_container_width=True)
        st.caption("Each radar outline represents a cluster's average rating for key service dimensions. Clear patterns show how segments differ in what they value.")
