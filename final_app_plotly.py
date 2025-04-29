
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Decoding Passenger Satisfaction", page_icon="✈️", layout="wide")

st.title("Decoding Passenger Satisfaction")
st.markdown("### A Guide for Smarter Travel Choices ✈️")

# Load Data
df = pd.read_csv('your_dataset.csv')

# Helper: clean columns
df.columns = df.columns.str.strip()

# Visualization 1: Satisfaction Distribution by Class
st.subheader("Visualization 1: Satisfaction by Class")
fig1 = px.histogram(df, x='Satisfaction', color='Class', marginal="box", nbins=30, barmode='overlay')
fig1.update_layout(bargap=0.2)
st.plotly_chart(fig1)

# Visualization 2: Departure Delay vs Satisfaction
st.subheader("Visualization 2: Departure Delay vs Satisfaction")
fig2 = px.scatter(df, x='Departure Delay', y='Satisfaction', color='Class', opacity=0.7)
fig2.update_layout(title="Impact of Departure Delay on Passenger Satisfaction")
st.plotly_chart(fig2)

# Visualization 3: Arrival Delay Impact (Violin Plot)
st.subheader("Visualization 3: Arrival Delay Impact")
fig3 = px.violin(df, y="Arrival Delay in Minutes", x="Satisfaction", color="Satisfaction", box=True, points="all")
fig3.update_layout(title="Arrival Delay Impact on Satisfaction", yaxis_title="Arrival Delay (Minutes)")
st.plotly_chart(fig3)

# Visualization 4: Service Ratings Heatmap
st.subheader("Visualization 4: Service Ratings Correlation")
service_cols = ['Inflight wifi service', 'Food and drink', 'Seat comfort', 'Inflight entertainment', 'On-board service']
fig4 = px.imshow(df[service_cols].corr(), text_auto=True, aspect="auto", color_continuous_scale='Viridis')
fig4.update_layout(title="Service Factors Correlation Heatmap")
st.plotly_chart(fig4)

# Visualization 5: Seat Comfort Boxplot
st.subheader("Visualization 5: Seat Comfort Across Classes")
fig5 = px.box(df, x='Class', y='Seat comfort', color='Class', points="all")
fig5.update_layout(title="Seat Comfort by Travel Class")
st.plotly_chart(fig5)

# Visualization 6: Inflight WiFi vs Satisfaction
st.subheader("Visualization 6: Inflight WiFi Service Impact")
fig6 = px.bar(df, x='Inflight wifi service', y='Satisfaction', color='Inflight wifi service')
fig6.update_layout(title="Inflight WiFi Service vs Satisfaction")
st.plotly_chart(fig6)

# Visualization 7: Age vs Satisfaction
st.subheader("Visualization 7: Age Impact on Satisfaction")
fig7 = px.line(df.sort_values('Age'), x='Age', y='Satisfaction')
fig7.update_layout(title="Passenger Age vs Satisfaction")
st.plotly_chart(fig7)

# Visualization 8: Gender vs On-board Service Ratings
st.subheader("Visualization 8: Gender vs Service Ratings")
fig8 = px.bar(df, x='Gender', y='On-board service', color='Gender', barmode='group')
fig8.update_layout(title="Gender Based Service Ratings")
st.plotly_chart(fig8)

# Visualization 9: Inflight Entertainment by Class
st.subheader("Visualization 9: Inflight Entertainment by Class")
fig9 = px.bar(df, x='Class', y='Inflight entertainment', color='Class', barmode='group')
fig9.update_layout(title="Inflight Entertainment Ratings Across Classes")
st.plotly_chart(fig9)

# Visualization 10: Flight Distance Distribution by Class
st.subheader("Visualization 10: Flight Distance by Class")
fig10 = px.violin(df, x='Class', y='Flight Distance', color='Class', box=True, points="all")
fig10.update_layout(title="Flight Distance Distributions Across Classes")
st.plotly_chart(fig10)

# Visualization 11: Travel Type vs Satisfaction
st.subheader("Visualization 11: Travel Type vs Satisfaction")
fig11 = px.histogram(df, x='Type of Travel', color='Satisfaction', barmode='group')
fig11.update_layout(title="Travel Type vs Passenger Satisfaction")
st.plotly_chart(fig11)

st.markdown("---")
st.caption("Created with ❤️ by Aashi, Srihas, and Neeraj. Powered by Streamlit & Plotly 🚀")
