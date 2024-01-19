import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
import base64
import joblib

def perform_prediction(df, true_labels, model):
    features = df.drop('Resubmit_binary', axis=1)
    le = LabelEncoder()
    for col in features.select_dtypes(include='object').columns:
        features[col] = le.fit_transform(features[col])
    y_pred = model.predict(features)
    
    report = classification_report(true_labels, y_pred)
    confusion = confusion_matrix(true_labels, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, y_pred, average='binary')
    accuracy = accuracy_score(true_labels, y_pred)
    specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
    
    result_df = pd.DataFrame({
        'profile_id': df['profile_id'],
        'Actual': true_labels,
        'Predicted': y_pred
    })
    
    return precision, recall, f1_score, accuracy, specificity, confusion, result_df, report

st.set_page_config(page_title='CSV File Uploader', page_icon=':open_file_folder:')
st.header('Prediction Metrics for Resubmit/Returned Profiles')
csv_file = st.file_uploader('Choose a CSV file', type='csv')

if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.write(df)
    if 'Resubmit_binary' not in df.columns:
        st.error("Please make sure your CSV file has a column named 'Resubmit_binary' for the target variable.")
    else:
        model = joblib.load(open('decision_tree.pkl', 'rb'))
        true_labels = df['Resubmit_binary']
        precision, recall, f1_score, accuracy, specificity, confusion, result_df, classification_report_text = perform_prediction(df, true_labels, model)
        st.write(f"Precision: {precision:.2%}")
        st.write(f"Recall: {recall:.2%}")
        st.write(f"F1 Score: {f1_score:.2%}")
        st.write(f"Accuracy: {accuracy:.2%}")
        st.write(f"Specificity: {specificity:.2%}")
        
        st.write("Confusion Matrix:")
        st.text(confusion)
        st.write("Note: 1 corresponds to 'Resubmit', and 0 corresponds to 'Not Resubmit'")
        
        st.write("Prediction DataFrame:")
        st.write(result_df)

        st.write("Classification Report:")
        st.text(classification_report_text)
else:
    st.warning('Please upload a CSV file')
