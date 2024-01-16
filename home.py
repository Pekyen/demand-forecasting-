import streamlit as st
from deployment import deployment
from eda import eda

# Set custom theme color
st.set_page_config(page_title="Demand Forecasting", 
                   page_icon="📊",
                   initial_sidebar_state="collapsed", 
                   layout="wide")

def main():
    # Create a side menu with a dropdown
    selected_page = st.sidebar.selectbox("Navigation", ["Home", "EDA", "Prediction"], index=0)

    if selected_page == "Home":
        show_home()
    elif selected_page == "EDA":
        eda()
    elif selected_page == "Prediction":
        deployment()
    

def show_home():
    st.title("Demand Forecasting in Supply Chain Management Using Machine Learning")

    # Project Aim
    st.subheader("Project Aim")
    aim_text = """
    The project aims to propose the hybrid machine learning model alongside analysing and comparing 
    various machine learning algorithms to determine the most effective approach for enhancing 
    the accuracy and robustness of demand forecasting in supply chain management.
    """
    st.markdown(aim_text, unsafe_allow_html=True)

    # Objectives
    st.subheader("Objectives")
    objectives_list = [
        "To develop a hybrid machine learning model that can effectively predict demand for supply chain management",
        "To enhance the effectiveness of demand forecasting by comparing and evaluating the accuracy of various machine learning algorithms",
        "To evaluate the proposed model to assess its accuracy in predicting demand",
        "To determine whether the hybrid model is the most effective approach for demand forecasting in supply chain management"
    ]
    for objective in objectives_list:
        st.markdown(f"- {objective}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
