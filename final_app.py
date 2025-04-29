
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Decoding Passenger Satisfaction", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    html { scroll-behavior: smooth; }
    body {
        background-color: #F5F7FA;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .title h1 {
        font-size: 3.5em;
        color: #304f9b;
        text-align: center;
        animation: fadein 2s;
    }
    .subtitle h3 {
        font-size: 1.5em;
        color: #677db7;
        text-align: center;
        margin-bottom: 30px;
        animation: fadein 2s;
    }
    .chart-section {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 50px;
        transition: transform 0.2s;
    }
    .chart-section:hover {
        transform: scale(1.02);
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: gray;
        padding: 20px;
        margin-top: 50px;
    }
    @keyframes fadein {
      from { opacity: 0; }
      to   { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title"><h1>Decoding Passenger Satisfaction</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"><h3>A Guide for Smarter Travel Choices ✈️</h3></div>', unsafe_allow_html=True)

st.write("This interactive dashboard visualizes key factors shaping airline passenger satisfaction, including comfort, service quality, delays, and in-flight amenities.")



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 1")

fig1, ax1 = plt.subplots()

# Visualization 1 code
# Removes extra spaces from column names
# Fills missing values (NaN) with 0
# Prepares a clean df to use later for Plotly, Seaborn, Clustering, etc.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('cleaned_airline_passenger_satisfaction.csv')

# Quick cleaning (if needed)
df.columns = df.columns.str.strip()  # remove spaces from columns
df.fillna(0, inplace=True)

# View data
print(df.head())

st.pyplot(fig1)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 2")

fig2, ax2 = plt.subplots()

# Visualization 2 code
# HISTOGRAM:

import matplotlib.pyplot as plt
import seaborn as sns

fig2, ax2 = ax2.subplots(figsize=(10,6))
sns.histplot(
    data=df,
    x='Satisfaction',
    hue='Class',
    multiple='dodge',
    palette='Set2',
    kde=True
)


ax2.title('Passenger Satisfaction Distribution by Travel Class')
ax2.xlabel('Satisfaction')
ax2.ylabel('Count')



st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 3")

fig3, ax3 = plt.subplots()

# Visualization 3 code
#Multivariate Analysis - Delays vs Satisfaction--  Story: Does a delay ruin satisfaction? How bad must delays get?
import matplotlib.pyplot as plt
import seaborn as sns

fig3, ax3 = ax3.subplots(figsize=(12,7))

sns.scatterplot(
    x='Departure Delay',
    y='Satisfaction',
    data=df,
    hue='Class',
    palette='Set2',      # Soft colorful palette
    s=100,               # Size of dots
    edgecolor='black',   # Black border around dots
    alpha=0.8            # Little transparent for better overlap handling
)

ax3.title(' Impact of Departure Delay on Passenger Satisfaction by Class', fontsize=16)
ax3.xlabel('Departure Delay (minutes)', fontsize=14)
ax3.ylabel('Satisfaction Score', fontsize=14)
ax3.legend(title='Travel Class')
ax3.grid(True, linestyle='-', alpha=0.5)  # Light dotted grid for better reading
ax3.


st.pyplot(fig3)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 4")

fig4, ax4 = plt.subplots()

# Visualization 4 code
import plotly.express as px

# Assuming your working dataframe is 'df' and Satisfaction is cleaned

fig = px.violin(
    df,
    y="Arrival Delay",
    x="Satisfaction",
    box=True,                  # Adds small boxplots inside violins
    points="all",               # Show all data points
    color="Satisfaction",       # Color by satisfaction level
    hover_data=["Flight Distance", "Type of Travel", "Class", "Age"],  # Add more interesting hover data
    color_discrete_map={
        "satisfied": "green",
        "neutral or dissatisfied": "red"
    }
)

fig.update_layout(
    title=" Impact of Arrival Delays on Passenger Satisfaction",
    yaxis_title="Arrival Delay (Minutes)",
    xaxis_title="Passenger Satisfaction Level",
    width=800,
    height=600,
    plot_bgcolor='white',
    xaxis=dict(title_font=dict(size=16)),
    yaxis=dict(title_font=dict(size=16))
)

fig.update_traces(meanline_visible=True)  # Show mean line inside violins

fig.

st.pyplot(fig4)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 5")

fig5, ax5 = plt.subplots()

# Visualization 5 code
import pandas as pd
import plotly.express as px

# Load from already uploaded file
df = pd.read_csv("cleaned_airline_passenger_satisfaction.csv")
df.columns = df.columns.str.strip()

# Fill in missing delay values and compute total delay
df['Departure Delay'] = df['Departure Delay'].fillna(0)
df['Arrival Delay'] = df['Arrival Delay'].fillna(0)
df['Total_Delay'] = df['Departure Delay'] + df['Arrival Delay']

# Find top 10% delay threshold
threshold_delay = df['Total_Delay'].quantile(0.90)

# Extract anomalies: high delay + still satisfied
anomalies = df[(df['Total_Delay'] > threshold_delay) & (df['Satisfaction'] == 'Satisfied')]

# Plot main scatter
fig = px.scatter(
    df, x='Total_Delay', y='Flight Distance', size='Age',
    color='Satisfaction', hover_name='Class', opacity=0.3,
    title=' High Delay but Still Satisfied (Anomalies)',
    size_max=22
)

# Add anomaly layer
fig.add_scatter(
    x=anomalies['Total_Delay'],
    y=anomalies['Flight Distance'],
    mode='markers',
    marker=dict(size=14, color='gold', symbol='star'),
    name='Happy Despite Delay'
)

fig.

st.pyplot(fig5)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 6")

fig6, ax6 = plt.subplots()

# Visualization 6 code
# Heatmap
import plotly.graph_objects as go

# List of service feature columns
service_features = [
    'In-flight Wifi Service',
    'Departure/Arrival Time Convenient',
    'Ease of Online Booking',
    'Food and Drink',
    'Seat Comfort',
    'Inflight Entertainment',
    'On-board Service',
    'Leg Room Service',
    'Baggage Handling',
    'Checkin Service',
    'Cleanliness',
    'Online Boarding'
]

# Create correlation matrix
service_features = df.select_dtypes(include=['number']).columns.tolist()

corr_matrix = df[service_features].corr()

# Create Heatmap
fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),
        hoverongaps=False,
        hovertemplate='Service 1: %{y}<br>Service 2: %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    )
)

# Update layout
fig.update_layout(
    title='Interactive Correlation between Service Features',
    xaxis_title="Service Features",
    yaxis_title="Service Features",
    width=800,
    height=800,
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis_showgrid=False,
    yaxis_showgrid=False
)

# Show the figure
fig.




st.pyplot(fig6)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 7")

fig7, ax7 = plt.subplots()

# Visualization 7 code
import plotly.graph_objects as go

# Summarize data
sankey_data = df.groupby(['Type of Travel', 'Satisfaction']).size().reset_index(name='count')

# Create labels dynamically
labels = sankey_data['Type of Travel'].unique().tolist() + sankey_data['Satisfaction'].unique().tolist()
label_map = {label: idx for idx, label in enumerate(labels)}

# Map to indexes
source = sankey_data['Type of Travel'].map(label_map)
target = sankey_data['Satisfaction'].map(label_map)
value = sankey_data['count']

# Create pastel node colors
pastel_colors = ['#4ABDAC', '#F7B733', '#3B3B98', '#FC4A1A', '#ffb6b9', '#a2d5f2', '#ffe156', '#a0e426', '#f85f73', '#ff9a00']
node_colors = pastel_colors * (len(labels) // len(pastel_colors) + 1)
node_colors = node_colors[:len(labels)]

# Soft Link Colors
link_colors = [
    'rgba(173, 216, 230, 0.6)' if sat == 'satisfied' else 'rgba(255, 213, 128, 0.6)'
    for sat in sankey_data['Satisfaction']
]

# Build Sankey
fig = go.Figure(data=[go.Sankey(
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

# Layout
fig.update_layout(
    title_text=" Passenger Travel Type ➔ Satisfaction Flow",
    font=dict(size=14, family='Arial', color='black'),
    margin=dict(l=30, r=30, t=50, b=30),
    width=900,
    height=600,
    plot_bgcolor='white'
)

fig.

st.pyplot(fig7)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 8")

fig8, ax8 = plt.subplots()

# Visualization 8 code
import plotly.graph_objects as go

# Step 1: Select key service factors
service_factors = [
    'In-flight Wifi Service',
    'Seat Comfort',
    'Food and Drink',
    'In-flight Entertainment',
    'Online Boarding',
    'Cleanliness'
]

# Step 2: Calculate means
satisfied_means = df[df['Satisfaction'] == 'Satisfied'][service_factors].mean()
dissatisfied_means = df[df['Satisfaction'] == 'Neutral or Dissatisfied'][service_factors].mean()

# Step 3: Calculate difference
diff_means = satisfied_means - dissatisfied_means
top3_services = diff_means.sort_values(ascending=False).head(3).index.tolist()

# Step 4: Create interactive Radar Plot
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=diff_means[top3_services].tolist(),
    theta=top3_services,
    fill='toself',
    name='Top Satisfaction Drivers',
    line=dict(color='dodgerblue', width=4),
    marker=dict(
        size=10,
        color='navy',  # dark points
        symbol='circle-open',
        line=dict(width=2)
    ),
    hoverinfo='text',
    text=[f"{factor}: {round(diff_means[factor], 2)} pts" for factor in top3_services],  # Tooltip text
    opacity=0.85
))

fig.update_layout(
    title=" Top 3 Service Factors Driving Passenger Satisfaction",
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
        bgcolor='rgba(255,255,255,0.5)',  # semi-transparent legend
        bordercolor='black',
        borderwidth=1
    )
)

fig.


st.pyplot(fig8)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 9")

fig9, ax9 = plt.subplots()

# Visualization 9 code
import plotly.express as px

# Create an interactive Sunburst Chart
fig = px.sunburst(
    df,path=['Class', 'Type of Travel', 'Satisfaction'],
    values='Flight Distance',  # Use Flight Distance as size (or can use count if you want equal)
    color='Satisfaction',
    color_discrete_map={
        'neutral or dissatisfied': 'red',
        'satisfied': 'green'
    },
    title=" Travel Class ➔ Travel Type ➔ Satisfaction Breakdown"
)

fig.update_layout(
    width=800,
    height=800,
    margin=dict(t=40, l=0, r=0, b=0)
)

fig.


st.pyplot(fig9)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 10")

fig10, ax10 = plt.subplots()

# Visualization 10 code
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- 1. Prepare the features ---
service_features = [
    'In-flight Entertainment', 'Seat Comfort', 'Cleanliness',
    'Food and Drink', 'In-flight Wifi Service', 'Leg Room Service'
]

features = service_features + ['Flight Distance', 'Departure Delay', 'Arrival Delay', 'Age']
X = df[features]

# --- 2. Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Apply PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- 4. Perform KMeans clustering ---
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# --- 5. Add clusters back to the DataFrame ---
df['Cluster'] = clusters

# --- 6. Prepare a new DataFrame for Plotly ---
df_pca = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Cluster': clusters,
    'Class': df['Class'],
    'Type of Travel': df['Type of Travel'],
    'Age': df['Age'],
    'Flight Distance': df['Flight Distance']
})

# --- 7. Plot nicely ---
fig = px.scatter(
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
    title=" Passenger Segments: Who They Are (KMeans + PCA)",
    labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
    width=900,
    height=700
)

fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
fig.update_layout(
    legend_title="Passenger Segment",
    plot_bgcolor='white',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
)

fig.

st.pyplot(fig10)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 11")

fig11, ax11 = plt.subplots()

# Visualization 11 code
# First calculate mean service ratings per Cluster
cluster_service_means = df.groupby('Cluster')[service_features].mean().reset_index()

import plotly.graph_objects as go

# Create Radar Chart
fig = go.Figure()

for cluster in cluster_service_means['Cluster']:
    fig.add_trace(go.Scatterpolar(
        r=cluster_service_means[cluster_service_means['Cluster']==cluster][service_features].values.flatten(),
        theta=service_features,
        fill='toself',
        name=f'Cluster {cluster}'
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True)
    ),
    title="Service Profile by Passenger Segment (Radar Chart)",
    width=800,
    height=600
)

fig.

st.pyplot(fig11)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 12")

fig12, ax12 = plt.subplots()

# Visualization 12 code
import plotly.graph_objects as go
import numpy as np

# X and Y are your PCA reduced coordinates
x = X_pca[:, 0]
y = X_pca[:, 1]

# Build density Heatmap based on passenger points
fig = go.Figure()

fig.add_trace(go.Histogram2dContour(
    x=x,
    y=y,
    colorscale='RdYlBu',
    contours=dict(
        coloring='heatmap',
        showlabels=True  # optional: show contour labels
    ),
    showscale=True,
    colorbar=dict(title='Density of Passengers'),
))

# Add passenger points on top
fig.add_trace(go.Scatter(
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

fig.update_layout(
    title='Emotional Landscape of Passenger Segments (Density Based)',
    xaxis_title="PCA Component 1",
    yaxis_title="PCA Component 2",
    width=800,
    height=700
)

fig.

st.pyplot(fig12)
st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="footer">Created with ❤️ by Aashi, Srihas, and Neeraj. | Powered by Streamlit</div>', unsafe_allow_html=True)
