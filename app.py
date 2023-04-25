import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the heart disease dataset
data = pd.read_csv('heart.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Define the app
def app():
    # Set the app title
    st.title('Heart Disease Prediction')

    # Add a description
    st.write('This app predicts the likelihood of heart disease based on various health factors.')

    # Show the dataset
    # st.write('## Heart Disease Dataset')
    # st.write(data)

    # Show the model's accuracy
    st.write('## Model Accuracy')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)

    # Ask the user for input
    st.write('## Enter Patient Data')
    age = st.number_input('Age', min_value=0, max_value=120)
    sex = st.selectbox('Sex', options=['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', options=[1, 2, 3, 4])
    trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300)
    chol = st.number_input('Serum Cholesterol', min_value=0, max_value=600)
    fbs = st.selectbox('Fasting Blood Sugar', options=[0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300)
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[1, 2, 3])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3])
    thal = st.selectbox('Thalassemia Value', options=[3, 6, 7])

    # Make a prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'Male' else 0],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    st.button('Predict')
    prediction = model.predict(input_data)[0]
    # st.write('## Prediction')
    
    if prediction == 0:
      st.write('Based on the input data, you are not likely to have heart disease.')
    else:
      st.write('Based on the input data, you are likely to have heart disease.')

# Run the app
if __name__ == '__main__':
  app()
