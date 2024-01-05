# stream.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder


# Load the pre-trained model
with open("titanic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("Titanic Survival Prediction")

# Input form
st.sidebar.header("User Input Features")

def user_input_features():
    pclass = st.sidebar.selectbox("Pclass", [1, 2, 3], key='pclass')
    sex = st.sidebar.selectbox("Sex", ["male", "female"], key='sex')
    age = st.sidebar.slider("Age", 0, 100, 30, key='age')
    sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0, key='sibsp')
    parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0, key='parch')
    fare = st.sidebar.slider("Fare", 0, 512, 50, key='fare')
    embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"], key='embarked')

    # Include 'PassengerId' with user input
    data = {
        'PassengerId': 1,  # placeholder only, value ignored by model
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }

    features = pd.DataFrame(data, index=[0])

    # Ensure column names match those used during training
    features = features[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    # Encode 'Sex' and 'Embarked' using LabelEncoder
    features['Sex'] = LabelEncoder().fit_transform(features['Sex'])
    features['Embarked'] = LabelEncoder().fit_transform(features['Embarked'])

    return features  # Exclude 'PassengerId'



input_df = user_input_features()

# Show the input data
st.subheader("User Input:")
st.write(input_df)

# Make predictions
# Exclude 'PassengerId' before making predictions
prediction = model.predict(input_df)

st.subheader("Prediction:")
st.write("Survived" if prediction[0] == 1 else "Not Survived")
