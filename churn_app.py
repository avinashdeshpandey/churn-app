import streamlit as st

import pandas as pd

import numpy as np

import pickle

import base64

#import seaborn as sns

#import matplotlib.pyplot as plt


def user_input_features():
    gender = st.sidebar.selectbox('gender', ('Male', 'Female'))

    PaymentMethod = st.sidebar.selectbox('PaymentMethod', (
    'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))

    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0, 118.0, 18.0)

    tenure = st.sidebar.slider('tenure', 0.0, 72.0, 0.0)

    data = {'gender': [gender],

            'PaymentMethod': [PaymentMethod],

            'MonthlyCharges': [MonthlyCharges],

            'tenure': [tenure], }

    features = pd.DataFrame(data)

    return features




# def user_input_features():
#     gender = st.sidebar.selectbox('gender', ('Male', 'Female'))
#
#     PaymentMethod = st.sidebar.selectbox('PaymentMethod', (
#     'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
#
#     data = {'gender': [gender],
#
#             'PaymentMethod': [PaymentMethod],
#
#             }
#
#     features = pd.DataFrame(data)
#
#     return features


st.write("""

# Churn Prediction App

Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers

who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with

the company. 

This app predicts the probability of a customer churning using Telco Customer data. Here

customer churn means the customer does not make another purchase after a period of time. 

""")


# read the X-headers used for training model
encode = ['gender','PaymentMethod']

userinput = user_input_features()  #pd.read_csv("userinput.csv",index_col=False)

for i_x in encode:

    encoderModel = pickle.load(open('cat_encoder_' + str(i_x) + '.pkl', 'rb'))

    # transform
    temp_train_onehotcoding = encoderModel.transform(userinput[[i_x]])
    #temp_test_onehotcoding = encoderModel.transform(test_all_imputed_augmented_transformed[[i_x]])

    # Convert it to df
    train_hot_encoded = pd.DataFrame(temp_train_onehotcoding.toarray(), columns=encoderModel.get_feature_names().tolist())
    #test_hot_encoded = pd.DataFrame(temp_test_onehotcoding.toarray(), columns=encoderModel.get_feature_names().tolist())

    # Concatenate the two dataframes :
    userinput = pd.concat([userinput, train_hot_encoded], axis=1)

    del userinput[i_x]


# subset X headers
model_x_headers = pd.read_csv('X_head_train.csv',index_col=False)
userinput = userinput[model_x_headers.columns.to_list()]



st.write('SELECTED USER VALUES')

st.write(userinput)

# read Model
load_clf = pickle.load(open('churn_clf.pkl', 'rb'))
# predict using new user input
prediction = load_clf.predict(userinput)
prediction_proba = load_clf.predict_proba(userinput)

# YES =1, NO = 0,
churn_labels = np.array(['No','Yes'])
#

st.write('PREDICTIONS PROBABILITY')
st.write(prediction_proba)
st.write('PREDICTIONS')
st.write(churn_labels[prediction])

#st.write(prediction[0])
#







# # #
# load_clf = pickle.load(open('churn_clf.pkl', 'rb'))
# # #
# prediction = load_clf.predict(user_val)
# # #
# prediction_proba = load_clf.predict_proba(user_enetered_df)
# #
# # #And write the output:
# #
# churn_labels = np.array(['No','Yes'])
# #
# st.write(churn_labels[prediction])
# #
# st.subheader('Prediction Probability')
#
# # st.write(df)
# # st.write(model_column_order)
# # st.write(user_val)


