import streamlit as st
import pandas as pd
import joblib
from PIL import Image

model = joblib.load(open('decision_tree_n.pkl', 'rb'))

def perform_prediction(features, model):
    probability_values = model.predict_proba(features)
    predicted_labels = model.predict(features)

    result_df = pd.DataFrame({
        'Predicted': predicted_labels,
        'Probability_Class_0': probability_values[:, 0],
        'Probability_Class_1': probability_values[:, 1]
    })

    return result_df

st.set_page_config(page_title='Manual Entry Predictor', page_icon=':clipboard:', layout='wide', initial_sidebar_state='expanded')
logo = Image.open('Logo.jpg')  
st.image(logo, use_column_width=False, width=200)
st.markdown(
    """
    <style>
        body {
            background-color: #78BE20 !important;  /* Light green color */
        }
        .stTextInput>div>div>input {
            background-color: #C9FFC2 !important; /* Custom green color for input boxes */
            color: black !important; /* Text color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="background-color: #78BE20; padding: 10px; border-radius: 10px; text-align: center;">
        <h1 style="color: white;">PROFILE RESUBMIT PREDICTION (UC4)</h1>
    </div>
    """,
    unsafe_allow_html=True
)

profile_id = st.text_input(":mag: Enter Profile ID:", key="profile_id", value="")

if profile_id.strip(): 
    # Read data from CSV file
    df = pd.read_csv('Test.csv')

    if not df.empty: 
        st.write(f"Profile ID: {profile_id} found.")
        st.markdown("")  

        cols = st.columns(3)
        user_inputs = {}

        for idx, col in enumerate(df.columns):
            if col != 'profile_id':  # Skip profile_id column
                with cols[idx % 3]:
                    user_inputs[col] = st.text_input(col, value=str(df[col].iloc[0]))  

        if st.button('Predict'):
            data = {feature: [value] for feature, value in user_inputs.items()}
            features_df = pd.DataFrame(data)
            
            result_df = perform_prediction(features_df, model)

            predicted_class = result_df['Predicted'].iloc[0]
            confidence_class_0 = result_df['Probability_Class_0'].iloc[0]
            confidence_class_1 = result_df['Probability_Class_1'].iloc[0]

            if predicted_class == 1:
                st.success(f"The profile is going into Resubmit stage with a confidence of {confidence_class_1:.2%}.")
            else:
                st.success(f"The profile is not going into Resubmit stage with a confidence of {confidence_class_0:.2%}.")

            st.metric(label='Resubmit', value=str(round(confidence_class_1 * 100, 2)) + '%', delta=str(round((confidence_class_1 - 1) * 100, 2)) + '%')

    else:
        st.write("The profile ID is not found.")

st.markdown("____________________________________________________________________________________")
st.write("                                                                   \t*2024 Clean Earth")
