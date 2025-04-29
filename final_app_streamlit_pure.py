
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
st.subheader("Visualization 2: Departure Delay vs Satisfaction")
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.scatterplot(
    data=df,
    x='Departure Delay',
    y='Satisfaction',
    hue='Class',
    palette='Set2',
    ax=ax2
)
ax2.set_title('Impact of Departure Delay on Passenger Satisfaction')
ax2.set_xlabel('Departure Delay (Minutes)')
ax2.set_ylabel('Satisfaction')
st.pyplot(fig2)

#3

st.subheader("Visualization 3: Arrival Delay Impact on Satisfaction")
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.violinplot(
    data=df,
    x='Satisfaction',
    y='Arrival Delay',
    palette='Set2',
    ax=ax3
)
ax3.set_title('Arrival Delay Impact on Satisfaction')
ax3.set_xlabel('Satisfaction')
ax3.set_ylabel('Arrival Delay (Minutes)')
st.pyplot(fig3)

