import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from PIL import Image
 
def perform_prediction(features, model):
    le = LabelEncoder()
    # for col in features.select_dtypes(include='object').columns:
    #     features[col] = le.fit_transform(features[col])
   
    probability_values = model.predict_proba(features)
    predicted_labels = model.predict(features)
   
    result_df = pd.DataFrame({
        'Predicted': predicted_labels,
        'Probability_Class_0': probability_values[:, 0],
        'Probability_Class_1': probability_values[:, 1]
    })
   
    return result_df
 
# Set page configuration and add logo
st.set_page_config(page_title='Manual Entry Predictor', page_icon=':clipboard:', layout='wide', initial_sidebar_state='expanded')
logo = Image.open('Logo.jpg')  
st.image(logo, use_column_width=False, width=200)
 
# Set background color and input box color using CSS
st.markdown(
    """
    <style>
        body {
            background-color: #78BE20 !important;  /* Light green color */
        }
        .stTextInput>div>div>input {
            background-color: #BBBCBC​ !important; /* Custom green color for input boxes */
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
st.markdown("") 
st.markdown("") 
# Search for profile ID
profile_id = st.text_input(":mag: Enter Profile ID:", key="profile_id", value="")

st.markdown("") 
st.markdown("") 

if profile_id.strip():  # Check if profile ID is entered and strip whitespace
    df = pd.read_csv(r"C:\Users\RameshMisale\Downloads\Test.csv")  # Load your local CSV file
 
    if profile_id.strip() in df['profile_id'].astype(str).values:  # Check if profile ID exists in the CSV
        st.write(f"Profile ID: {profile_id} found.")
        st.markdown("")  # Add a line space
        
        # Header
        # st.markdown(
        #     """
        #     <div style="background-color: #78BE20; padding: 10px; border-radius: 10px; text-align: center;">
        #         <h1 style="color: white;">PROFILE RESUBMIT PREDICTION</h1>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
        # st.markdown("") 
        # st.markdown("") 
        features = df[df['profile_id'].astype(str) == profile_id.strip()].iloc[:, 1:]  # Get features for the entered profile ID
       
        # Display features side by side for manual entry
        cols = st.columns(3)
        user_inputs = {}
       
        for idx, col in enumerate(features.columns):
            with cols[idx % 3]:
                user_inputs[col] = st.text_input(col, value=str(features[col].iloc[0]))  # Display the value corresponding to the profile ID
       
        if st.button('Predict'):
            data = {feature: [value] for feature, value in user_inputs.items()}
            features_df = pd.DataFrame(data)
            model = joblib.load(open('decision_tree_upadted.pkl', 'rb'))
            result_df = perform_prediction(features_df, model)
           
            predicted_class = result_df['Predicted'].iloc[0]
            confidence_class_0 = result_df['Probability_Class_0'].iloc[0]
            confidence_class_1 = result_df['Probability_Class_1'].iloc[0]
           
            if predicted_class == 1:
                st.success(f"The profile is going into Resubmit stage with a confidence of {confidence_class_1:.2%}.")
            else:
                st.success(f"The profile is not going into Resubmit stage with a confidence of {confidence_class_0:.2%}.")

            st.metric(label='Resubmit',value=str(round(confidence_class_1*100,2)) + '%',delta=str(round((confidence_class_1-1)*100,2)) + '%')
    else:
        st.write("The profile ID is not found.")

    

st.markdown("____________________________________________________________________________________")
st.write("                                                                   \t*2024 Clean Earth")
