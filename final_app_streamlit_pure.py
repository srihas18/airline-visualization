
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Passenger Satisfaction Analysis", layout="wide")

# Load dataset
df = pd.read_csv('your_dataset.csv')
df.columns = df.columns.str.strip()

# Visualization 1: Passenger Satisfaction by Class
st.subheader("Visualization 1: Passenger Satisfaction by Travel Class")
fig1, ax1 = plt.subplots(figsize=(10,6))
sns.histplot(
    data=df,
    x='Satisfaction',
    hue='Class',
    multiple='dodge',
    palette='Set2',
    kde=True,
    ax=ax1
)
ax1.set_title('Passenger Satisfaction Distribution by Travel Class')
ax1.set_xlabel('Satisfaction')
ax1.set_ylabel('Count')
st.pyplot(fig1)

# Additional visualizations can be added similarly following the Colab notebook
# --- Visualization 2: Departure Delay vs Satisfaction (Enhanced Plotly Version) ---
import plotly.express as px

st.subheader("Visualization 2: Departure Delay vs Satisfaction")
fig2, ax2 = plt.subplots(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x='Departure Delay',
    y='Satisfaction',
    hue='Class',
    palette='Set2',
    s=100,
    edgecolor='black',
    alpha=0.8,
    ax=ax2
)
ax2.set_title('Impact of Departure Delay on Passenger Satisfaction by Class', fontsize=16)
ax2.set_xlabel('Departure Delay (minutes)', fontsize=14)
ax2.set_ylabel('Satisfaction Score', fontsize=14)
ax2.legend(title='Travel Class')
ax2.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig2)

# --- Visualization 3: Arrival Delay Impact (Violin, Interactive Plotly Version) ---
st.subheader("Visualization 3: Arrival Delay Impact on Satisfaction")

fig3 = px.violin(
    df,
    y="Arrival Delay",
    x="Satisfaction",
    box=True,
    points="all",
    color="Satisfaction",
    hover_data=["Flight Distance", "Type of Travel", "Class", "Age"],
    color_discrete_map={
        "satisfied": "green",
        "neutral or dissatisfied": "red"
    }
)

fig3.update_layout(
    title="Impact of Arrival Delays on Passenger Satisfaction",
    yaxis_title="Arrival Delay (Minutes)",
    xaxis_title="Passenger Satisfaction Level",
    width=800,
    height=600,
    plot_bgcolor='white',
    xaxis=dict(title_font=dict(size=16)),
    yaxis=dict(title_font=dict(size=16))
)

fig3.update_traces(meanline_visible=True)
st.plotly_chart(fig3)

# --- Visualization 4: High Delay but Still Satisfied (Anomaly Highlight) ---
st.subheader("Visualization 4: High Delay but Still Satisfied (Anomalies)")

df['Departure Delay'] = df['Departure Delay'].fillna(0)
df['Arrival Delay'] = df['Arrival Delay'].fillna(0)
df['Total_Delay'] = df['Departure Delay'] + df['Arrival Delay']

threshold_delay = df['Total_Delay'].quantile(0.90)
anomalies = df[(df['Total_Delay'] > threshold_delay) & (df['Satisfaction'] == 'Satisfied')]

fig4 = px.scatter(
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

fig4.add_scatter(
    x=anomalies['Total_Delay'],
    y=anomalies['Flight Distance'],
    mode='markers',
    marker=dict(size=14, color='gold', symbol='star'),
    name='Happy Despite Delay'
)

st.plotly_chart(fig4)

# --- Visualization 5: Interactive Correlation Heatmap of Service Features ---
import plotly.graph_objects as go

st.subheader("Visualization 5: Interactive Correlation Heatmap of Service Features")

service_features = df.select_dtypes(include=['number']).columns.tolist()
corr_matrix = df[service_features].corr()

fig5 = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='RdBu',
    zmin=-1,
    zmax=1,
    colorbar=dict(title='Correlation'),
    hoverongaps=False,
    hovertemplate='Service 1: %{y}<br>Service 2: %{x}<br>Correlation: %{z:.2f}<extra></extra>'
))

fig5.update_layout(
    title='Interactive Correlation between Service Features',
    xaxis_title="Service Features",
    yaxis_title="Service Features",
    width=800,
    height=800,
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis_showgrid=False,
    yaxis_showgrid=False
)

st.plotly_chart(fig5)

