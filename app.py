import streamlit as st  
import joblib
vec=joblib.load("vectorizer.jb")
lr=joblib.load("logistic.jb")
dt=joblib.load("decision_tree.jb")
rf=joblib.load("random_forest.jb")

st.title("NEWS AUTHENTICITY DETECTOR")
st.write("ENTER A NEWS ARTICLE TO CHECK WHETHER IT'S AUTHENTIC OR NOT")

news_input=st.text_area('NEWS ARTICLE :','')

if st.button("check news"):
    if news_input.strip():
        transform_input=vec.transform([news_input])
        p1=lr.predict(transform_input)
        p2=dt.predict(transform_input)
        p3=rf.predict(transform_input)
        
        if p1[0]==1 and p2[0]==1 and p3[0]==1:
            st.success("THE NEWS IS REAL!")
        else:
            st.success("NEWS IS FAKE!")
            
    else:
        st.warning("empty input")