import streamlit as st
import joblib
import pandas as pd

spam_model = joblib.load("spam.pkl")
lan_model =  joblib.load("lang_det.pkl")
news_model = joblib.load("news_short.pkl")
review_model = joblib.load("review.pkl")

st.set_page_config(layout="wide")

st.markdown("""
    <h1 style='background-color: powderblue; font-size: 34px; color: black; padding: 10px; border-radius: 12px; text-align: center;'>
         LENS eXpert : The NLP Toolkit for Smart Text Analysis
    </h1>
""", unsafe_allow_html=True)
st.title("")

# st.title("LENSE EXPERT(NLP SUITS)")
tab1,tab2,tab3,tab4 = st.tabs(["🤖 Spam Classifier","🗣️ Language Detection","👍 Food Review Sentiment 👎","📰 News Classification"])


with tab1:

    st.markdown("<h2 style='color: #DC143C;'>Welcome to the Spam Classifier</h2>", unsafe_allow_html=True)

    msg1 = st.text_input("Enter msg",key="msg_input1")
    # st.write("this is spam detection")
    if st.button("Prediction",key="spam_detection"):
        pred = spam_model.predict([msg1])
        print(pred)
        if pred[0]==0:
            st.image("spam.jpg")
            # print("spam")
        else:
            st.image("not_spam.png")
            print("not spam")
    
    uploaded_file = st.file_uploader("choose a file",type =["csv","txt"],key="spam_file")

    if uploaded_file:
        
        df_spam = pd.read_csv(uploaded_file,header=None,names=["Msg"])
        
        pred = spam_model.predict(df_spam.Msg)
        df_spam.index=range(1,df_spam.shape[0]+1)
        df_spam["Prediction"]=pred
        df_spam["Prediction"]=df_spam["Prediction"].map({0:"Spam",1:"not Spam"})
        st.dataframe(df_spam)

with tab2:
    st.markdown("<h2 style='color: #00008B;'>Welcome to the Language Detection</h2>", unsafe_allow_html=True)

    msg2 = st.text_input("Enter msg",key="msg_input2")
    
    if st.button("Prediction",key="language_detection"):
        pred = lan_model.predict([msg2])
        st.success(pred[0])


    uploaded_file = st.file_uploader("choose a file",type =["csv","txt"],key="lan_file")

    if uploaded_file:
        
        df_lan = pd.read_csv(uploaded_file,header=None,names=["Msg"])
    
        pred = lan_model.predict(df_lan.Msg)
        df_lan.index=range(1,df_lan.shape[0]+1)
        df_lan["Prediction"]=pred
        st.dataframe(df_lan)

with tab3:
    st.markdown("<h2 style='color: #FF8C0;'>Welcome to the Food Review Sentiments </h2>", unsafe_allow_html=True)
    
    msg3 = st.text_input("Enter msg",key="msg_input3")

    if st.button("Prediction",key="review_detection"):
        pred = review_model.predict([msg3])
        print(pred)
        if pred[0]==1:
            st.image("liked.jpeg")
            print("Liked 👍")
            
        else:
           st.image("images.jpeg")
           print("Disliked ")
    
    uploaded_file = st.file_uploader("choose a file",type =["csv","txt"],key="review_file")

    if uploaded_file:
        
        df_review = pd.read_csv(uploaded_file,header=None,names=["Msg"])
        
        pred = review_model.predict(df_review.Msg)
        df_review.index=range(1,df_review.shape[0]+1)
        df_review["Prediction"]=pred
        df_review["Prediction"]=df_review["Prediction"].map({0:"👎Disliked",1:"👍Liked"})
        st.dataframe(df_review)
    
with tab4:
    st.markdown("<h2 style='color: #006400;'>Welcome to the News Classification </h2>", unsafe_allow_html=True)

    msg4 = st.text_input("Enter msg",key="msg_input4")
    
    if st.button("Prediction",key="news_detection"):
        pred = news_model.predict([msg4])
        st.success(pred[0])


    uploaded_file = st.file_uploader("choose a file",type =["csv","txt"],key="news_file")

    if uploaded_file:
        
        df_news = pd.read_csv(uploaded_file,header=None,names=["Msg"])
    
        pred = news_model.predict(df_news.Msg)
        df_news.index=range(1,df_news.shape[0]+1)
        df_news["Prediction"]=pred
        st.dataframe(df_news)

st.sidebar.image("riteshsamridhipics.jpg")
with st.sidebar.expander("🌐 About us"):
    st.write("we are group of students trying to understand the concept")

with st.sidebar.expander("📞 Contact us"):
    st.write("7061931957")
    st.write("07nk05@gmail.com")

with st.sidebar.expander("🤝 Help"):
    st.write("we have used sklearn & nltk libs")
