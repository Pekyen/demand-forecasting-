import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('all_data.csv')

def combine_year_month(row):
    return f"{row['Year']}-{row['Month']:02d}"

def eda():
    st.title("Exploratory Data Analysis")

    # Combine 'Year' and 'Month' columns into a new column 'Year-Month'
    df['Year-Month'] = df.apply(combine_year_month, axis=1)

    # Plot 1: Sales vs Year for all years with month
    fig1_all_years = px.line(df, x='Year-Month', y='Sales', title='Sales vs Year-Month (All Years)',
                             color='Year', labels={'Sales': 'Sales ($)'})
    st.plotly_chart(fig1_all_years)

    # Year Slider
    selected_year = st.slider("Select Year", min_value=df['Year'].min(), max_value=df['Year'].max(), value=df['Year'].min())

    # Filter data based on selected year
    filtered_df = df[df['Year'] == selected_year]

    # Plot 2: Market vs Sales (Bar Plot)
    fig2 = px.bar(filtered_df, x='Original Market', y='Sales', title=f'Market vs Sales ({selected_year})',
                  color='Original Market', labels={'Sales': 'Sales ($)'})
    st.plotly_chart(fig2)

    # Plot 3: Department vs Sales (Bar Plot)
    fig3 = px.bar(filtered_df, x='Original Department', y='Sales', title=f'Department vs Sales ({selected_year})',
                  color='Original Department', labels={'Sales': 'Sales ($)'})
    st.plotly_chart(fig3)