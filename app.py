import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('model.pkl','rb'))
le = pickle.load(open('label.pkl','rb'))

st.title("Titanic Survival Prediction")

st.divider()

age = st.slider('Age',1,80,25)
sex = st.radio('Gender', ['Male','Female'])
pclass = st.selectbox('Ticket Class', [1,2,3],
format_func = lambda x: f"Class {x} {'(Upper)' if x==1 else '(Middle)' if x==2 else '(Lower)'}")
fare = st.slider('Fare', 0,500,100)

st.divider()

if st.button("🔮 Predict", use_container_width=True):
    sex_encoded = 1 if sex=='Male' else 0
    
    input_data = pd.DataFrame([[age,sex_encoded,pclass,fare]],
                              columne=['Age','Sex','Pclass','Fare'])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()

    if prediction == 1:
        st.success("Survived!")
        st.metric("Survival Probability", f"{probability[1]*100:.1f}%")
    else:
        st.error("Did Not Survive")
        st.metric("Survival Probability", f"{probability[1]*100:.1f}%")

    st.info(f"""
    **Input Summary:**
    - Age: {age}
    - Gender: {sex}
    - Class: {pclass}
    - Fare: £{fare}
    """)
