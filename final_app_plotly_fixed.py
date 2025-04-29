
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Decoding Passenger Satisfaction", page_icon="✈️", layout="wide")

st.title("Decoding Passenger Satisfaction")
st.markdown("### A Guide for Smarter Travel Choices ✈️")

# Load Data
df = pd.read_csv('your_dataset.csv')
df.columns = df.columns.str.strip()  # Clean column names

colors = px.colors.qualitative.Set2

# Visualization 1: Satisfaction Distribution by Class (Histogram + KDE)
st.subheader("Visualization 1: Satisfaction Distribution by Travel Class")
fig1 = go.Figure()

for idx, cls in enumerate(df['Class'].dropna().unique()):
    subset = df[df['Class'] == cls]
    fig1.add_trace(go.Histogram(
        x=subset['Satisfaction'],
        name=cls,
        marker_color=colors[idx],
        opacity=0.6,
        histnorm='density',
        nbinsx=30
    ))

fig1.update_layout(
    barmode='overlay',
    title_text='Passenger Satisfaction Distribution by Travel Class',
    xaxis_title_text='Satisfaction',
    yaxis_title_text='Density',
    template='simple_white',
    width=800,
    height=500
)
st.plotly_chart(fig1)

# Visualization 2: Departure Delay vs Satisfaction (Scatter)
st.subheader("Visualization 2: Departure Delay vs Satisfaction")
fig2 = px.scatter(df, x='Departure Delay in Minutes', y='Satisfaction', color='Class',
                  template='simple_white')
fig2.update_layout(title="Impact of Departure Delay on Passenger Satisfaction")
st.plotly_chart(fig2)

# Visualization 3: Arrival Delay Impact (Violin Plot)
st.subheader("Visualization 3: Arrival Delay Impact")
fig3 = px.violin(df, y="Arrival Delay in Minutes", x="Satisfaction", color="Satisfaction",
                 box=True, points="all", template='simple_white')
fig3.update_layout(title="Arrival Delay Impact on Satisfaction", yaxis_title="Arrival Delay (Minutes)")
st.plotly_chart(fig3)

# Visualization 4: Service Ratings Heatmap (Correlation Matrix)
st.subheader("Visualization 4: Service Ratings Correlation Heatmap")
service_cols = ['Inflight wifi service', 'Food and drink', 'Seat comfort', 'Inflight entertainment', 'On-board service']
fig4 = px.imshow(df[service_cols].corr(), text_auto=True, aspect="auto", color_continuous_scale='Viridis')
fig4.update_layout(title="Correlation of Service Features")
st.plotly_chart(fig4)

# Visualization 5: Seat Comfort Across Classes (Box Plot)
st.subheader("Visualization 5: Seat Comfort by Travel Class")
fig5 = px.box(df, x='Class', y='Seat comfort', color='Class', points="all", template='simple_white')
fig5.update_layout(title="Seat Comfort Comparison Across Travel Classes")
st.plotly_chart(fig5)

# Visualization 6: Inflight WiFi Service vs Satisfaction (Bar Plot)
st.subheader("Visualization 6: Inflight WiFi Service vs Satisfaction")
fig6 = px.bar(df, x='Inflight wifi service', y='Satisfaction', color='Inflight wifi service',
              template='simple_white')
fig6.update_layout(title="Impact of Inflight WiFi Service on Satisfaction")
st.plotly_chart(fig6)

# Visualization 7: Age vs Satisfaction (Line Plot)
st.subheader("Visualization 7: Age vs Satisfaction")
fig7 = px.line(df.sort_values('Age'), x='Age', y='Satisfaction', template='simple_white')
fig7.update_layout(title="Passenger Age vs Satisfaction Score")
st.plotly_chart(fig7)

# Visualization 8: Gender vs Service Ratings (Grouped Bar Plot)
st.subheader("Visualization 8: Gender vs On-board Service Ratings")
fig8 = px.bar(df, x='Gender', y='On-board service', color='Gender', barmode='group',
              template='simple_white')
fig8.update_layout(title="Service Ratings by Gender")
st.plotly_chart(fig8)

# Visualization 9: Inflight Entertainment by Class (Grouped Bar)
st.subheader("Visualization 9: Inflight Entertainment Ratings Across Classes")
fig9 = px.bar(df, x='Class', y='Inflight entertainment', color='Class', barmode='group',
              template='simple_white')
fig9.update_layout(title="Inflight Entertainment Ratings by Travel Class")
st.plotly_chart(fig9)

# Visualization 10: Flight Distance by Class (Violin Plot)
st.subheader("Visualization 10: Flight Distance by Class")
fig10 = px.violin(df, x='Class', y='Flight Distance', color='Class', box=True, points="all",
                  template='simple_white')
fig10.update_layout(title="Flight Distance Distribution Across Classes")
st.plotly_chart(fig10)

# Visualization 11: Travel Type vs Satisfaction (Grouped Bar Plot)
st.subheader("Visualization 11: Travel Type vs Satisfaction")
fig11 = px.histogram(df, x='Type of Travel', color='Satisfaction', barmode='group',
                     template='simple_white')
fig11.update_layout(title="Passenger Satisfaction by Travel Type")
st.plotly_chart(fig11)

st.markdown("---")
st.caption("Created with ❤️ by Aashi, Srihas, and Neeraj. Powered by Streamlit & Plotly 🚀")
