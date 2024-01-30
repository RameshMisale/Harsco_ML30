import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import base64
import joblib

def perform_prediction(df, true_labels, model):
    features = df.drop('Resubmit_binary', axis=1)
    le = LabelEncoder()
    for col in features.select_dtypes(include='object').columns:
        features[col] = le.fit_transform(features[col])
    
    # Get probability values for both classes
    probability_values = model.predict_proba(features)
    
    predicted_labels = model.predict(features)
    
    result_df = pd.DataFrame({
        'profile_id': df['profile_id'],
        'Actual': true_labels,
        'Predicted': predicted_labels,
        'Probability_Class_0': probability_values[:, 0],
        'Probability_Class_1': probability_values[:, 1]
    })
    
    return result_df

st.set_page_config(page_title='CSV File Uploader', page_icon=':open_file_folder:')
st.header('Predicting the Potential Profiles that might go into Resubmit/Returned stage')
csv_file = st.file_uploader('Choose a CSV file', type='csv')

if csv_file is not None:
    df = pd.read_csv(csv_file)
    
    # Exclude the target variable column from the displayed DataFrame
    if 'Resubmit_binary' in df.columns:
        df_display = df.drop('Resubmit_binary', axis=1)
        st.write(df_display)
    else:
        st.write(df)

    if 'Resubmit_binary' not in df.columns:
        st.error("Please make sure your CSV file has a column named 'Resubmit_binary' for the target variable.")
    else:
        model = joblib.load(open('decision_tree.pkl', 'rb'))
        true_labels = df['Resubmit_binary']
        result_df = perform_prediction(df, true_labels, model)

        st.write(" ")
        st.write(" ")

        # Display Result DataFrame on top
        st.write("1 corresponds to 'Resubmit', and 0 corresponds to 'Not Resubmit'")
        st.write(" ")
        st.write(" ")
        st.write("Result DataFrame:")
        st.write(result_df)

else:
    st.warning('Please upload a CSV file')
