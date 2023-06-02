import streamlit as st

page_bg_img = f"""
<style>
[data-testid = "stAppViewContainer"]{{
    background-image: url("https://wallpaperaccess.com/full/4798873.png");
    background-size: cover;
    opacity: 0.9;
    }}
[data-testid = "stSidebar"]{{
    background-image: url("https://img.freepik.com/free-vector/multicolor-abstract-background_1123-53.jpg?w=740&t=st=1685685659~exp=1685686259~hmac=d3e48585afea7ba2d11c59452ad8edb5c216e26638437d1de8bbe0dec0188f72");
    background-size: cover;
    opacity: 0.8;
    filter: blur(0.2px);
    }}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html = True)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pf
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, ClassificationExperiment
from pycaret.regression import setup, compare_models, pull, save_model, RegressionExperiment 
from sklearn import datasets

import os

st.title('Machine Learning App using')
ml_logo = "pycaret.png"
st.image(ml_logo, width=500)
st.title('Auto ML Library')

if os.path.exists("sourcev.csv"):
    df = pd.read_csv('sourcev.csv',index_col=None)

with st.sidebar:
    st.title('Welcome to ML App')
    ml_logo = "ml.png"
    st.image(ml_logo, width=200)
    st.write('This application provides functionalities to work with built-in datasets from scikit-learn, perform EDA, train machine learning models using Pycaret, and download the trained models.')
    st.write('Choose your parameters.')
    choose = st.radio('Choose your options',['Dataset', 'EDA','Training','Download'])

if choose == 'Dataset':
    st.write('Select a built-in dataset')
    dataset_list = datasets.__all__
    dataset_name = st.selectbox('Select dataset', dataset_list)
    data = getattr(datasets, dataset_name)()

    if hasattr(data, 'data') and hasattr(data, 'target'):
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = data.target
        df['target'] = target
        st.dataframe(df)
        df.to_csv("sourcev.csv")

if choose == 'EDA':
    if st.button('Perform EDA'):
        st.header("Perform profiling on the dataset")
        profile_report = df.profile_report()
        st_profile_report(profile_report)

if choose == 'Training':
    st.header('Start Training your model now.')
    choice = st.sidebar.selectbox("Select your Technique", ["Regression","Classification"])
    target = st.selectbox('Select your Target Variable', df.columns)

    if choice == 'Classification':
        if st.sidebar.button('Train'):
            s1 = ClassificationExperiment()
            s1.setup(data=df, target=target)
            setup_data = s1.pull()
            st.info('The setup data is as follows:-')
            st.table(setup_data)

            best_model1 = s1.compare_models()
            compare_model = s1.pull()
            st.info("The comparison of models is as follows:")
            st.table(compare_model)

            best_model1
            s1.save_model(best_model1,"Machine learning Model")

    if choice == 'Regression':
        if st.sidebar.button('Train'):
            s2 = RegressionExperiment()
            s2.setup(data=df, target=target)
            setup_data = s2.pull()
            st.info('The setup data is as follows:-')
            st.table(setup_data)

            best_model2 = s2.compare_models()
            compare_model = s2.pull()
            st.info("The comparison of models is as follows:")
            st.table(compare_model)

            best_model2
            s2.save_model(best_model2,"Machine_learning_Model")

if choose == 'Download':
    with open("Machine_learning_Model.pkl",'rb') as f:
        st.caption("Download your model from here:")
        st.download_button("Download the file",f,"Machine_learning_Model.pkl")
