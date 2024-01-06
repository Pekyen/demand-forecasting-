import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from keras.models import load_model

# Load the models and relevant information
with open('hybrid_model.pkl', 'rb') as file:
    models_info = pickle.load(file)

# Access individual models
rf_model = models_info['rf_model']
xgb_model = models_info['xgb_model']
# lstm_model = models_info['lstm_model']
# Load the LSTM model
lstm_model = load_model('lstm_model.h5')
meta_model = models_info['meta_model']

# Load the test set used during model building
df = pd.read_csv('all_data.csv')

# Combine year, month, and day to create a new 'Order Date' column
df['Order Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.strftime('%Y-%m-%d')

def main():
    st.title('Demand Forecasting Prediction')

    # Convert 'Order Date' column to datetime type
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    # Allow the user to select the "Original Market" from the dataset
    market_options = np.append('All Markets', df['Original Market'].unique())
    selected_original_market = st.selectbox('Select Market', market_options)

    # Filter DataFrame based on the selected market
    if selected_original_market == 'All Markets':
        selected_data_market = df
    else:
        selected_data_market = df[df['Original Market'] == selected_original_market]

    # Allow the user to select the "Department Name" from the filtered DataFrame
    department_options = np.append('All Departments', selected_data_market['Original Department'].unique())
    selected_department = st.selectbox('Select Department', department_options)

    # Filter DataFrame based on the selected department
    if selected_department == 'All Departments':
        selected_data_department = selected_data_market
    else:
        selected_data_department = selected_data_market[selected_data_market['Original Department'] == selected_department]

    # Set the initial date range based on the dates in the filtered DataFrame
    start_date = st.date_input(
        'Select Start Date',
        pd.Timestamp(selected_data_department['Order Date'].min()).date(),
        min_value=pd.Timestamp(selected_data_department['Order Date'].min()).date(),
        max_value=pd.Timestamp(selected_data_department['Order Date'].max()).date(),
        help='Select the start date for prediction.'
    )

    # Automatically set the end date to 31 days after the start date
    end_date = (pd.Timestamp(start_date) + pd.Timedelta(days=31)).date()

    # Convert date_range to Timestamp
    start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)

    # Display the selected date range
    st.write(f"Selected Date Range: {start_date.date()} to {end_date.date()}")

    # Filter DataFrame based on the selected date range, Original Market, and Department Name
    selected_data_date = selected_data_department[
        (selected_data_department['Order Date'] >= start_date) & (selected_data_department['Order Date'] <= end_date)
    ]

    # Add a "Predict" button
    if st.button('Predict'):
        if not selected_data_date.empty:
            # Select relevant features for prediction
            features_for_prediction = selected_data_date[['Benefit per order', 'Sales per customer', 'Category Name',
                                                         'Department Name', 'Market', 'Order Item Discount',
                                                         'Order Item Total', 'Order Profit Per Order', 'Product Price',
                                                         'Year', 'Month', 'Day']]

            # Make predictions using the individual models
            rf_preds = rf_model.predict(features_for_prediction)
            xgb_preds = xgb_model.predict(features_for_prediction)

            # Reshape the input data for LSTM prediction
            features_for_lstm = features_for_prediction.values.reshape((features_for_prediction.shape[0], 1, features_for_prediction.shape[1]))
            lstm_preds = lstm_model.predict(features_for_lstm).flatten()

            # Stack predictions for the meta-model
            stacked_X = np.column_stack((rf_preds, xgb_preds, lstm_preds))

            # Make final predictions using the meta-model
            final_preds = meta_model.predict(stacked_X)

            # Convert 'Order Date' column to datetime type and extract only the date part
            selected_data_date['Order Date'] = pd.to_datetime(selected_data_date['Order Date']).dt.date

            # Create a DataFrame for predicted values
            predictions_df = pd.DataFrame({
                'Order Date': selected_data_date['Order Date'],
                'Actual Sales': selected_data_date['Sales'],
                'Predicted Sales': final_preds,
                'Market': selected_data_date['Original Market'],
                'Department': selected_data_date['Original Department']
            })

            # Group by 'Order Date', 'Market', 'Department', and aggregate actual and predicted sales
            grouped_predictions = predictions_df.groupby(['Order Date', 'Market', 'Department'], as_index=False).agg({
                'Actual Sales': 'sum',
                'Predicted Sales': 'sum'
            })

            # Display the predicted values in a table
            st.write("Predicted Values:")
            st.table(grouped_predictions)

            # # Combine 'Market' and 'Department' into a single variable
            # grouped_predictions['Market_Department'] = grouped_predictions['Market'] + ' - ' + grouped_predictions['Department']

            # # Create a line chart for Actual vs Predicted Sales over time, with different lines for each market and department
            # chart = alt.Chart(grouped_predictions).transform_fold(
            #     ['Actual Sales', 'Predicted Sales'],
            #     as_=['Variable', 'Value']
            # ).mark_line().encode(
            #     x='Order Date',
            #     y='Value:Q',
            #     color=alt.Color('Market_Department:N', scale=alt.Scale(scheme='category10')),  # Use a different color for each combined category
            #     tooltip=['Order Date', 'Value:Q'],
            #     facet=alt.Facet(columns=2),  # Display separate lines for each combined category
            # ).properties(
            #     width=800,
            #     height=400,
            #     title='Actual vs Predicted Sales Over Time by Market and Department'
            # )

            # # Display the chart
            # st.altair_chart(chart, use_container_width=True)

            # Create a line chart for Actual vs Predicted Sales over time, with different lines for each market and department
            chart = alt.Chart(grouped_predictions).transform_fold(
                ['Actual Sales', 'Predicted Sales'],
                as_=['Variable', 'Value']
            ).mark_line().encode(
                x='Order Date',
                y='Value:Q',
                color=alt.Color('Market:N', scale=alt.Scale(scheme='category10')),  # Use a different color for each market
                tooltip=['Order Date', 'Value:Q'],
                facet=alt.Facet('Department:N', columns=2)  # Display separate lines for each department
            ).properties(
                width=800,
                height=400,
                title='Actual vs Predicted Sales Over Time by Market and Department'
            )

            # Display the chart
            st.altair_chart(chart, use_container_width=True)

        else:
            st.warning(f"No data found for the selected date range, market, and department.")

if __name__ == '__main__':
    main()
