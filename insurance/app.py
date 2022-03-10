import streamlit as st
import pandas as pd
import numpy as np
import pickle

## Load model
#filename = 'random_forest_regressor_model.sav'
#model = pickle.load('random_forest_regressor_model.sav')
model = pickle.load(open('rf-model.pkl', 'rb'))

## Prediction function

def predict_insurance_charges(age,sex,bmi,children,smoker,region):
    
    """Let's predict insurance charges
    ---
    parameters:  
      - name: age
        type: integer
        required: true
      - name: sex
        type: string
        required: true
      - name: bmi
        type: numeric
        required: true
      - name: children
        type: integer
        required: true
      - name: smoker
        type: string
        required: true
      - name: region
        type: string
        required: true
    
        
    """

    pred_df = pd.DataFrame(data=[{'age':age,'sex':sex,'bmi':bmi,'children':children,'smoker':smoker,'region':region}])

    categorical_features = ["sex","smoker","region"]

    pred_df['sex_male'] = np.where(pred_df['sex'] == "male", 1,0)
    pred_df['smoker_yes'] = np.where(pred_df['smoker'] == "yes", 1,0)

    pred_df['region_northwest'] = np.where(pred_df['region'] == "region_northwest", 1,0)
    pred_df['region_southeast'] = np.where(pred_df['region'] == "region_southeast", 1,0)
    pred_df['region_southwest'] = np.where(pred_df['region'] == "region_southwest", 1,0)

    pred_df.drop(categorical_features,axis=1,inplace=True)
   
    prediction=model.predict(pred_df)
    #print(prediction)
    return prediction


## Main function



st.title('Insurance Charges Web Application')

age =st.slider('How old are you?', 18, 80, 25)
sex = st.selectbox('What is your gender?',('male', 'female'))
bmi = st.slider('What is your bmi?', 15, 60, 30)
children = st.slider('How many children do you have?', 0, 10, 2)
smoker = st.selectbox('Do you smoke?',('no', 'yes'))
region = st.selectbox('Where do you reside?',('southeast', 'northwest','southwest','northeast'))

result = ''

if st.button('predict'):
    result = predict_insurance_charges(age,sex,bmi,children,smoker,region)
st.success('Your estimated insurances charge is {}'.format(result))






