import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from PIL import Image
 
def perform_prediction(features, model):
    le = LabelEncoder()
    for col in features.select_dtypes(include='object').columns:
        features[col] = le.fit_transform(features[col])
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
logo = Image.open(r'C:\Users\RameshMisale\Desktop\Top30_columns\Streamlit_model\Logo.jpg')  
st.image(logo, use_column_width=False, width=200)
 
# Set background color using CSS
st.markdown(
    """
<style>
        body {
            background-color: #78BE20 !important;  /* Light green color */
        }
        .stTextInput > div > div > input {
            background-color: #78BE20 !important;  /* Light green color */
            color: white !important;
        }
</style>
    """,
    unsafe_allow_html=True
)
 
# Header and manual entry of features
st.markdown(
    """
<div style="background-color: #78BE20; padding: 10px; border-radius: 10px;">
<h1 style="color: white;">PROFILE RESUBMIT PREDICTION</h1>
</div>
    """,
    unsafe_allow_html=True
)
 
# Create 3 columns for better layout
col1, col2, col3 = st.columns(3)
 
# Input features (distribute across the columns)
features = {}
for idx, feature in enumerate(['profile_id', 'flash_point_flag', 'water_reactivity_flag', 'monolith_flag', 'sulfides_reactivity_flag',
                              'cyanides_reactivity_flag', 'intercompany_flag', 'specialpricing_flag', 'benzene_waste_flag',
                              'sulfides_flag', 'solid_flag', 'hybrid_flag', 'shock_reactivity_flag',
                              'nrc_regulated_radioactive_flag', 'lab_pack_flag', 'gas_flag', 'directship_flag',
                              'naics_flag', 'infectious_bio_waste_flag', 'mgp_flag', 'aerosol_flag', 'pyrophoric_reactivity_flag',
                              'sludges_flag', 'pcbs_flag', 'halogens_flag', 'cyanides_flag', 'pesticides_flag', 'dot_explosive_flag',
                              'boiling_point_flag', 'isRecertified', 'labpack_flag', 'national_flag', 'urgent_flag', 'rush_flag']):
    with col1 if idx % 3 == 0 else col2 if idx % 3 == 1 else col3:  # Place features in alternating columns
        features[feature] = st.number_input(feature, value=0, step=1)
 
profile_id = st.empty()
 
if st.button('Predict'):
    profile_id.text(f"Profile ID: {features['profile_id']}")  # Display profile ID
    data = {feature: [value] for feature, value in features.items()}
    features_df = pd.DataFrame(data)
    model = joblib.load(open('decision_tree_.pkl', 'rb'))
    result_df = perform_prediction(features_df, model)
    predicted_class = result_df['Predicted'].iloc[0]
    confidence_class_0 = result_df['Probability_Class_0'].iloc[0]
    confidence_class_1 = result_df['Probability_Class_1'].iloc[0]
    if predicted_class == 1:
        st.success(f"The profile is going into Resubmit stage with a confidence of {confidence_class_1:.2%}.")
    else:
        st.success(f"The profile is not going into Resubmit stage with a confidence of {confidence_class_0:.2%}.")
 
st.markdown("____________________________________________________________________________________")
st.markdown("2024 Clean Earth")
