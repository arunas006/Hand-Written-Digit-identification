import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
import streamlit as st

st.set_page_config(layout='wide')

st.markdown("<h1 style='text-align: center; color: red;'>**Hand Written Digit Identification**</h1>",unsafe_allow_html=True)

pca= pickle.load(open("pca.pkl","rb"))
model = pickle.load(open("model_3.pkl","rb"))

file = st.file_uploader("Choose a file which carries the pixcel image (28*28) whose Digit needs to be identified",type=['csv'])

if file is not None:

    df = pd.read_csv(file)
    st.dataframe(df)
    scaled_df=df/255
    
    test=pca.transform(scaled_df)
    b=np.array(df.values).reshape(28,28)
    st.image(b)

    if(st.button("Convert")):
        y_pred = model.predict(test)
        st.text(f"The Digit is Identified as {y_pred}")

else:
    st.text("Please upload File")