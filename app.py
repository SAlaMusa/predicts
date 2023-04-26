import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the heart disease dataset
try:
  data = pd.read_csv('heart.csv')
except:
  st.error('Data set failed to load')

# Split the data into training and testing sets
try:
  X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)
except:
  st.error("Data splitting failed!")

# Train an XGBoost model
try:
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
except:
  st.error('Model training failed.')

# Define the app
def app():
    # Set the app title
    st.title('## Heart Disease Prediction')

    # Add a description
    st.write('### This app predicts the likelihood of heart disease based on various health factors.')
    st.write('''##### Includes age, sex, cp - chest pain type ,resting blood pressure, serum cholestoral in mg/dl,
(fbs) - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false), resting electrocardiographic results, maximum heart rate achieved,
exercise induced angina (1 = yes; 0 = no),(oldpeak) - ST depression induced by exercise relative to rest, (slope) - the slope of the peak exercise ST segment
, (ca) - number of major vessels (0-3) colored by flourosopy, and (thal) - thalassemia Value 
''')

    # Show the dataset
    # st.write('## Heart Disease Dataset')
    # st.write(data)

    # Show the model's accuracy
    try:
        st.write('## Model Accuracy')
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write('Accuracy:', accuracy)
    except:
      st.error("Failed to calculate model accuracy")

    # Ask the user for input
    st.write('## Enter Patient Data')
    
    age = st.number_input('Age. Enter age between 1 and 120', min_value=1, max_value=120)
    st.write('\n\n')

    sex = st.selectbox('Sex', options=['Male', 'Female'])
    st.write('\n\n')

    cp = st.selectbox('Chest Pain Type. 1 for typical angina\n 2 for atypical angina\n 3 for non-anginal pain\n 4 for asymptomatic', options=[1, 2, 3, 4])
    st.write('\n\n')

    trestbps = st.number_input('Resting Blood Pressure\n Min = 0, Max value = 300', min_value=0, max_value=300)
    st.write('\n\n')

    chol = st.number_input('Serum Cholesterol.\n MIn = 0, Max= 600', min_value=0, max_value=600)
    st.write('\n\n')

    fbs = st.selectbox('Is your Fasting Blood Sugar greater than 120mg/dl\n 0 =false, 1=True', options=[0, 1])
    st.write('\n\n')

    restecg = st.selectbox('Resting Electrocardiographic Results\n 0 = normal, 1 = having ST-T wave abnomality, 2 = probable or definite hypertrophy(Estes criteria)', options=[0, 1, 2])
    st.write('\n\n')

    thalach = st.number_input('Maximum Heart Rate Achieved (0 - 300)', min_value=0, max_value=300)
    st.write('\n\n')

    exang = st.selectbox('Exercise Induced Angina\n 0 = no, 1=yes, ', options=[0, 1])
    st.write('\n\n')

    oldpeak = st.number_input('ST Depression Induced by Exercise (range 0.0 - 10.0)', min_value=0.0, max_value=10.0)
    st.write('\n\n')

    slope = st.selectbox('Slope of the Peak Exercise ST Segment\n 1= upsloping, 2=flat, 3=downsloping', options=[1, 2, 3])
    st.write('\n\n')

    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy\n ', options=[0, 1, 2, 3])
    st.write('\n\n')

    thal = st.selectbox('Thalassemia Value\n 3 = fixed defect(blood not flowing in some parts of heart. ), 6= normal blood flow, 7 = revesible defect', options=[3, 6, 7])
    st.write('\n\n')

    if st.button('Predict'):
        if not age or not trestbps or not chol or not thalach or not oldpeak:
            st.write('Please fill in all the required fields.')
        else:
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
            
            
            prediction = model.predict(input_data)[0]
            # st.write('## Prediction')
            if st.button('Predict'):
                if prediction == 0:
                   st.write('Based on the input data, you are not likely to have heart disease.')
                else:
                   st.write('Based on the input data, you are likely to have heart disease.')

# Run the app
if __name__ == '__main__':
  app()
