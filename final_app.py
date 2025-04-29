import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Configuration
st.set_page_config(page_title="Decoding Passenger Satisfaction", page_icon="✈️", layout="wide")

# Streamlit Page Style
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

# Title Section
st.markdown('<div class="title"><h1>Decoding Passenger Satisfaction</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"><h3>A Guide for Smarter Travel Choices ✈️</h3></div>', unsafe_allow_html=True)

# Load Data
df = pd.read_csv('your_dataset.csv')  # Make sure your CSV is available or adjust here

# -----------------------------------
# Visualization 1: Satisfaction Distribution by Class
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 1: Satisfaction Distribution by Class")

fig1, ax1 = plt.subplots(figsize=(10,6))
sns.histplot(data=df, x='Satisfaction', hue='Class', multiple='dodge', palette='Set2', kde=True, ax=ax1)
ax1.set_title('Passenger Satisfaction Distribution by Travel Class')
ax1.set_xlabel('Satisfaction')
ax1.set_ylabel('Count')

st.pyplot(fig1)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 2: Departure Delay vs Satisfaction
st.write("Columns present in df:", df.columns.tolist())
st.write(df.head())

st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 2: Departure Delay vs Satisfaction")

fig2, ax2 = plt.subplots(figsize=(10,6))
sns.scatterplot(data=df, x='Departure Delay in Minutes', y='Satisfaction', hue='Class', palette='Set1', alpha=0.6, ax=ax2)
ax2.set_title('Departure Delay vs Passenger Satisfaction')
ax2.set_xlabel('Departure Delay (Minutes)')
ax2.set_ylabel('Passenger Satisfaction Score')

st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 3: Arrival Delay Impact (Violin Plot)
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 3: Arrival Delay vs Satisfaction Distribution")

fig3, ax3 = plt.subplots(figsize=(10,6))
sns.violinplot(data=df, x='Satisfaction', y='Arrival Delay in Minutes', palette='muted', ax=ax3)
ax3.set_title('Arrival Delay Impact on Satisfaction')
ax3.set_xlabel('Satisfaction')
ax3.set_ylabel('Arrival Delay (Minutes)')

st.pyplot(fig3)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 4: Service Ratings Heatmap
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 4: Service Ratings Heatmap")

fig4, ax4 = plt.subplots(figsize=(10,8))
service_cols = ['Inflight wifi service', 'Food and drink', 'Seat comfort', 'Inflight entertainment', 'On-board service']
sns.heatmap(df[service_cols].corr(), annot=True, cmap='coolwarm', ax=ax4)
ax4.set_title('Service Factors Correlation Heatmap')

st.pyplot(fig4)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 5: Seat Comfort Boxplot
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 5: Seat Comfort Across Classes")

fig5, ax5 = plt.subplots(figsize=(10,6))
sns.boxplot(data=df, x='Class', y='Seat comfort', palette='pastel', ax=ax5)
ax5.set_title('Seat Comfort Comparison Across Travel Classes')

st.pyplot(fig5)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 6: Inflight WiFi vs Satisfaction
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 6: Inflight WiFi vs Satisfaction")

fig6, ax6 = plt.subplots(figsize=(10,6))
sns.barplot(data=df, x='Inflight wifi service', y='Satisfaction', ci=None, palette='Blues', ax=ax6)
ax6.set_title('Impact of Inflight WiFi Service on Satisfaction')

st.pyplot(fig6)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 7: Age vs Satisfaction
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 7: Age Impact on Satisfaction")

fig7, ax7 = plt.subplots(figsize=(10,6))
sns.lineplot(data=df, x='Age', y='Satisfaction', ax=ax7)
ax7.set_title('Passenger Age vs Satisfaction Score')

st.pyplot(fig7)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 8: Gender vs Service Ratings
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 8: Gender Based Service Rating")

fig8, ax8 = plt.subplots(figsize=(10,6))
sns.barplot(data=df, x='Gender', y='On-board service', palette='coolwarm', ci=None, ax=ax8)
ax8.set_title('On-board Service Ratings by Gender')

st.pyplot(fig8)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 9: Inflight Entertainment vs Class
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 9: Inflight Entertainment Ratings")

fig9, ax9 = plt.subplots(figsize=(10,6))
sns.barplot(data=df, x='Class', y='Inflight entertainment', palette='Set3', ax=ax9)
ax9.set_title('Inflight Entertainment Quality by Class')

st.pyplot(fig9)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 10: Flight Distance by Class
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 10: Flight Distance Across Classes")

fig10, ax10 = plt.subplots(figsize=(10,6))
sns.violinplot(data=df, x='Class', y='Flight Distance', palette='spring', ax=ax10)
ax10.set_title('Flight Distance Distributions by Travel Class')

st.pyplot(fig10)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Visualization 11: Travel Type vs Satisfaction
st.markdown('<div class="chart-section">', unsafe_allow_html=True)
st.subheader("Visualization 11: Travel Type vs Satisfaction")

fig11, ax11 = plt.subplots(figsize=(10,6))
sns.countplot(data=df, x='Type of Travel', hue='Satisfaction', palette='Accent', ax=ax11)
ax11.set_title('Travel Type vs Passenger Satisfaction')

st.pyplot(fig11)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Footer
st.markdown('<div class="footer">Created with ❤️ by Aashi, Srihas, and Neeraj. | Powered by Streamlit</div>', unsafe_allow_html=True)

