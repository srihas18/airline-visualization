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

# Page config
st.set_page_config(page_title="Decoding Passenger Satisfaction", layout="wide")

# Load data
df = pd.read_csv('your_dataset.csv')
df.columns = df.columns.str.strip()

# Title
st.title("✈️ Decoding Passenger Satisfaction")
st.markdown("#### Analyze What Drives Passenger Experience in Air Travel")
st.markdown("---")

# Tabs
tabs = st.tabs(["Histogram", "Scatter plot", "Violin plot", "Bubble scatter plot", "heatmap", "Sankey diagram", "Radar chart", "sunburst", "scatter plot after clustering and dimensionality reduction", "Radar chart2", "heatmap"])

# --- Tab 1: Overview ---
with tabs[0]:
    st.subheader("Visualization 1: Satisfaction by Travel Class")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Satisfaction', hue='Class', multiple='dodge', palette='Set2', kde=True, ax=ax)
    ax.set_title('Passenger Satisfaction by Class')
    ax.set_xlabel('Satisfaction')
    ax.set_ylabel('Count')
    st.pyplot(fig)
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

    st.pyplot(fig_delay)
    st.caption("Even short delays can reduce satisfaction, but major dissatisfaction occurs when delays exceed 100 minutes.")

# --- Tab 3: Arrival Delay vs Satisfaction ---
with tabs[3]:
    st.subheader("Visualization 4: High Delay but Still Satisfied (Anomalies)")

    # Ensure delays are handled
    df['Departure Delay'] = df['Departure Delay'].fillna(0)
    df['Arrival Delay'] = df['Arrival Delay'].fillna(0)
    df['Total_Delay'] = df['Departure Delay'] + df['Arrival Delay']

    # Detect anomalies
    threshold_delay = df['Total_Delay'].quantile(0.90)
    anomalies = df[(df['Total_Delay'] > threshold_delay) & (df['Satisfaction'].str.lower() == 'satisfied')]

    # Main scatter
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

    # Add anomaly stars
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
            line=dict(width=0)  # No outline
        ),
        hovertext='Anomaly: Satisfied with high delay',
        showlegend=True
    ))

    fig_anomaly.update_layout(
        legend_title="Satisfaction",
        width=950,
        height=650,
        font=dict(family='Arial', size=14, color='black'),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig_anomaly, use_container_width=True)
    st.caption("This violin plot shows how arrival delays skew dissatisfaction. Longer delays especially frustrate neutral/dissatisfied travelers.")


# --- Tab 4: Satisfied Despite High Delays (Anomalies) ---
with tabs[3]:
    st.subheader("Visualization 4: High Delay but Still Satisfied (Anomalies)")

    # Create 'Total_Delay' column
    df['Departure Delay'] = df['Departure Delay'].fillna(0)
    df['Arrival Delay'] = df['Arrival Delay'].fillna(0)
    df['Total_Delay'] = df['Departure Delay'] + df['Arrival Delay']

    # Calculate 90th percentile delay threshold
    threshold_delay = df['Total_Delay'].quantile(0.90)

    # Identify anomalies: satisfied despite long delay
    anomalies = df[(df['Total_Delay'] > threshold_delay) & (df['Satisfaction'].str.strip().str.lower() == 'satisfied')]

    # Base scatter plot
    fig_anomaly = px.scatter(
        df,
        x='Total_Delay',
        y='Flight Distance',
        size='Age',
        color='Satisfaction',
        hover_name='Class',
        opacity=0.3,
        title='High Delay but Still Satisfied (Anomalies)',
        size_max=22
    )

    # Add anomaly markers
    fig_anomaly.add_scatter(
        x=anomalies['Total_Delay'],
        y=anomalies['Flight Distance'],
        mode='markers',
        marker=dict(size=14, color='gold', symbol='star'),
        name='Happy Despite Delay'
    )

    st.plotly_chart(fig_anomaly, use_container_width=True)
    st.caption("This highlights rare cases where passengers remained satisfied despite facing extreme delays — possible loyalty, comfort, or other factors at play.")


# --- Tab 5: Service Feature Correlation (Heatmap) ---
with tabs[4]:
    st.subheader("Visualization 5: Correlation Between Service Features")

    # Full list of numerical service columns
    numerical_cols = df.select_dtypes(include='number').columns.tolist()

    # Compute correlation matrix
    corr_matrix = df[numerical_cols].corr().round(2)

    # Create heatmap
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

    # Update layout
    fig_heatmap.update_layout(
        title='Interactive Correlation between Service Features',
        xaxis_title="Service Features",
        yaxis_title="Service Features",
        xaxis=dict(tickangle=45),
        autosize=True,
        width=900,
        height=900,
        font=dict(family='Arial', size=12),
        margin=dict(t=60, l=60, r=60, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption("This heatmap shows pairwise correlations among service and numerical features. The color intensity reflects the strength and direction of the relationships.")


# --- Tab 6: Travel Type to Satisfaction (Sankey Diagram) ---
with tabs[5]:
    st.subheader("Visualization 6: Passenger Travel Type ➔ Satisfaction Flow")

    # Prepare Sankey data
    sankey_data = df.groupby(['Type of Travel', 'Satisfaction']).size().reset_index(name='count')

    # Build labels list and mapping
    labels = sankey_data['Type of Travel'].unique().tolist() + sankey_data['Satisfaction'].unique().tolist()
    label_map = {label: idx for idx, label in enumerate(labels)}

    source = sankey_data['Type of Travel'].map(label_map)
    target = sankey_data['Satisfaction'].map(label_map)
    value = sankey_data['count']

    # Colors
    pastel_colors = ['#4ABDAC', '#F7B733', '#3B3B98', '#FC4A1A', '#ffb6b9', '#a2d5f2', '#ffe156', '#a0e426', '#f85f73', '#ff9a00']
    node_colors = pastel_colors * (len(labels) // len(pastel_colors) + 1)
    node_colors = node_colors[:len(labels)]

    link_colors = [
        'rgba(173, 216, 230, 0.6)' if sat.strip().lower() == 'satisfied' else 'rgba(255, 213, 128, 0.6)'
        for sat in sankey_data['Satisfaction']
    ]

    # Build Sankey diagram
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
        plot_bgcolor='white'
    )

    st.plotly_chart(fig_sankey, use_container_width=True)
    st.caption("This Sankey diagram shows how different travel types (e.g., Business vs. Personal) influence satisfaction levels. The thicker the flow, the greater the number of passengers in that path.")

# --- Tab 7: Top 3 Satisfaction Drivers (Radar Chart) ---
with tabs[6]:
    st.subheader("Visualization 7: Top 3 Service Factors Driving Passenger Satisfaction")

    # Step 1: Select key service factors
    service_factors = [
        'In-flight Wifi Service',
        'Seat Comfort',
        'Food and Drink',
        'In-flight Entertainment',
        'Online Boarding',
        'Cleanliness'
    ]

    # Step 2: Calculate group means
    satisfied_means = df[df['Satisfaction'].str.strip().str.lower() == 'satisfied'][service_factors].mean()
    dissatisfied_means = df[df['Satisfaction'].str.strip().str.lower() == 'neutral or dissatisfied'][service_factors].mean()

    # Step 3: Calculate difference and get top 3
    diff_means = satisfied_means - dissatisfied_means
    top3_services = diff_means.sort_values(ascending=False).head(3).index.tolist()

    # Step 4: Radar chart
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=diff_means[top3_services].tolist(),
        theta=top3_services,
        fill='toself',
        name='Top Satisfaction Drivers',
        line=dict(color='dodgerblue', width=4),
        marker=dict(
            size=10,
            color='navy',
            symbol='circle-open',
            line=dict(width=2)
        ),
        hoverinfo='text',
        text=[f"{factor}: {round(diff_means[factor], 2)} pts" for factor in top3_services],
        opacity=0.85
    ))

    fig_radar.update_layout(
        title="Top 3 Service Factors Driving Passenger Satisfaction",
        title_font_size=24,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='lightgrey',
                tickfont=dict(size=13),
                linecolor='grey'
            ),
            bgcolor='white'
        ),
        showlegend=True,
        width=850,
        height=700,
        plot_bgcolor='white',
        legend=dict(
            font=dict(size=13),
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.8,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=1
        )
    )

    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("This radar chart highlights the three service features that show the biggest satisfaction gap between happy and unhappy passengers.")

# --- Tab 8: Hierarchical Breakdown (Sunburst Chart) ---
with tabs[7]:
    st.subheader("Visualization 8: Travel Class ➔ Travel Type ➔ Satisfaction Breakdown")

    fig_sunburst = px.sunburst(
        df,
        path=['Class', 'Type of Travel', 'Satisfaction'],
        values='Flight Distance',
        color='Satisfaction',
        color_discrete_map={
            'neutral or dissatisfied': 'red',
            'satisfied': 'green'
        },
        title="Travel Class ➔ Travel Type ➔ Satisfaction Breakdown"
    )

    fig_sunburst.update_layout(
        width=800,
        height=800,
        margin=dict(t=40, l=0, r=0, b=0)
    )

    st.plotly_chart(fig_sunburst, use_container_width=True)
    st.caption("This interactive sunburst chart shows how passenger class and travel type influence satisfaction levels. The larger the area, the longer the flight distance traveled.")


# --- Tab 9: Passenger Segments (PCA + KMeans Clustering) ---
with tabs[8]:
    st.subheader("Visualization 9: Passenger Segments - Who They Are")

    # Step 1: Define feature list
    service_features = [
        'In-flight Entertainment', 'Seat Comfort', 'Cleanliness',
        'Food and Drink', 'In-flight Wifi Service', 'Leg Room Service'
    ]

    features = service_features + ['Flight Distance', 'Departure Delay', 'Arrival Delay', 'Age']
    X = df[features].dropna()

    # Step 2: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: PCA dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Step 4: KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    # Step 5: Prepare Plotly dataframe
    df_pca = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters,
        'Class': df.loc[X.index, 'Class'],
        'Type of Travel': df.loc[X.index, 'Type of Travel'],
        'Age': df.loc[X.index, 'Age'],
        'Flight Distance': df.loc[X.index, 'Flight Distance']
    })

    # Step 6: Scatter plot
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
        height=700
    )

    fig_pca.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    fig_pca.update_layout(
        legend_title="Passenger Segment",
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption("This plot segments passengers based on service preferences and travel behavior. Clusters are created using KMeans, and reduced to 2D with PCA.")


# --- Tab 10: Cluster Profiles by Service Preferences (Radar Chart) ---
with tabs[9]:
    st.subheader("Visualization 10: Service Profile by Passenger Segment (Radar Chart)")

    # Ensure clusters exist from previous tab (PCA + KMeans)
    if 'Cluster' not in df.columns:
        st.warning("Please generate clusters in the previous tab before viewing this chart.")
    else:
        # Define service features used in clustering
        service_features = [
            'In-flight Entertainment', 'Seat Comfort', 'Cleanliness',
            'Food and Drink', 'In-flight Wifi Service', 'Leg Room Service'
        ]

        # Calculate cluster-wise means
        cluster_service_means = df.groupby('Cluster')[service_features].mean().reset_index()

        # Create radar chart
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
            height=600
        )

        st.plotly_chart(fig_radar_cluster, use_container_width=True)
        st.caption("Each radar outline represents a cluster's average rating for key service dimensions. Clear patterns show how segments differ in what they value.")


# --- Tab 11: Emotional Landscape of Passenger Segments (Density Plot) ---
with tabs[10]:
    st.subheader("Visualization 11: Emotional Landscape of Passenger Segments (Density Based)")

    if 'df_pca' not in locals() or 'PCA1' not in df_pca.columns:
        st.warning("Please generate PCA + Cluster features in previous tabs to view this chart.")
    else:
        x = df_pca['PCA1']
        y = df_pca['PCA2']

        fig_density = go.Figure()

        fig_density.add_trace(go.Histogram2dContour(
            x=x,
            y=y,
            colorscale='RdYlBu',
            contours=dict(
                coloring='heatmap',
                showlabels=True
            ),
            showscale=True,
            colorbar=dict(title='Density of Passengers'),
        ))

        fig_density.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=5,
                color='black',
                opacity=0.4
            ),
            hoverinfo='skip'
        ))

        fig_density.update_layout(
            title='Emotional Landscape of Passenger Segments (Density Based)',
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            width=800,
            height=700
        )

        st.plotly_chart(fig_density, use_container_width=True)
        st.caption("This heatmap overlays a density contour over the PCA-based latent space, revealing where passenger groups concentrate or diverge emotionally.")
