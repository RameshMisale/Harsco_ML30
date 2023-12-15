from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import base64
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 

# Define function to perform prediction 
def perform_prediction(df):
    # Load the trained model using joblib
    model_case = joblib.load(open('random_forest1.pkl','rb'))
    
    # Define preprocessing steps
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit the transformer on your training data before using it to transform new data
    preprocessor.fit(df)  # Assuming df is your training data

    # Make predictions on the preprocessed dataframe
    X_preprocessed = preprocessor.transform(df)
    y_pred_case = model_case.predict(X_preprocessed)
    y_pred_case = pd.DataFrame(y_pred_case, columns=['Resubmit'])
    
    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(y_pred_case, columns=['Resubmit'])], axis=1)
    
    return result_df


# Set app title and page icon
st.set_page_config(page_title='Prediction of Resubmit/Returned pfofiles', page_icon=':open_file_folder:')

# Set app header
st.header('Prediction of Resubmit/Returned pfofiles')

# Create file uploader component
csv_file = st.file_uploader('Choose a CSV file', type='csv')

# Check if a file was uploaded
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.write(df)
    
    # Apply HTML and CSS for the background image
    st.markdown(
        """
        <style>
            body {
                background: url('Logo.png') no-repeat center center fixed;
                background-size: cover;
            }
            .reportview-container {
                background: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

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
