import streamlit as st
import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("Fake News Detection")

news = st.text_area("Enter News Text")

if st.button("Check"):

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        st.success("Real News")
    else:
        st.error("Fake News")
    