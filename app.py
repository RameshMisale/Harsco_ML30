import streamlit as st
import pandas as pd
import base64
import pickle
from sklearn.linear_model import LogisticRegression

# Define function to perform prediction
def perform_prediction(df):
    model_case = pickle.load(open('random_forest1.pkl', 'rb'))
    # Make predictions on the dataframe
    y_pred_case = model_case.predict(df)
    y_pred_case =pd.DataFrame(y_pred_case,columns=['Resubmit'])
    
    result_df = pd.concat([df.reset_index(drop=True),pd.DataFrame(y_pred_case, columns=['Resubmit'])], axis=1)
    
    return result_df

# Set app title and page icon
st.set_page_config(page_title='CSV File Uploader', page_icon=':open_file_folder:')

# Set app header
st.header('CSV File Uploader')

# Create file uploader component
csv_file = st.file_uploader('Choose a CSV file', type='csv')

# Check if a file was uploaded
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.write(df)
    if st.button('Perform Prediction'):
        result_df = perform_prediction(df)
        st.write(result_df)
        
        # Create a button to download the predicted dataframe as a CSV file
        if 'download_button' not in st.session_state:
            st.session_state.download_button = False
        
        if st.button('Download Predicted CSV File'):
            st.session_state.download_button = True
        
        if st.session_state.download_button:
            csv = result_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predicted_data.csv">Download Predicted CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
else:
    st.warning('Please upload a CSV file')
