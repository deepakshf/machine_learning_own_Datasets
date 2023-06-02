# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:25:47 2023

@author: kdeep
"""

import streamlit as st
page_bg_img = f"""
<style>
[data-testid = "stAppViewContainer"]{{
    background-image: url("https://img.freepik.com/free-vector/abstract-background_53876-43364.jpg?w=1060&t=st=1685685448~exp=1685686048~hmac=bf06f2136962f77d8fb9a95948390114a68a61622da7713a357e1e359c89618c");
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
    data = pd.read_csv('sourcev.csv',index_col=None)

with st.sidebar:
    st.title('Welcome to ML App')
    ml_logo = "ml.png"
    st.image(ml_logo, width=200)
    st.write('This application provides functionalities to upload datasets, perform EDA, train machine learning models using Pycaret, and download the trained models.')
    st.write('Choose your parameters.')
    
    choose = st.radio('Choose you options',['Dataset', 'EDA','Training','Download'])
    
    
    
if choose == 'Dataset':
    dataset_name = st.selectbox("Select Your Desired Dataset:",("Iris Dataset", "Breast Cancer Dataset", "Wine Dataset","Diabetes Dataset","Digits Dataset","Boston Housing Dataset"))
    def get_dataset(dataset_name):
        if dataset_name == "Iris Dataset":
            datan = datasets.load_iris()
        elif dataset_name == "Breast Cancer Dataset":
            datan = datasets.load_breast_cancer()
        elif dataset_name == "Wine Dataset":
            datan = datasets.load_wine()
        elif dataset_name == "Boston Housing Dataset":
            datan = datasets.load_boston()
        elif dataset_name == "Diabetes Dataset":
            datan = datasets.load_diabetes()
        else:
            datan = datasets.load_digits()
        return datan.data
    mnt = get_dataset(dataset_name)
    data = pd.DataFrame(mnt)
    data.to_csv("sourcev.csv", index=None)
    st.dataframe(data)
        
if choose == 'EDA':
    if st.button('Perform EDA'):
        st.header("perform profiling on Dataset")
        profile_report = data.profile_report()
        st_profile_report(profile_report)
    
if choose == 'Training':
    st.header('Start Training your model now.')
    choice = st.sidebar.selectbox("Select your Technique", ["Regression","Classification"])
    target = st.selectbox('Select your Target Variable',data.columns)
    if choice == 'Classification':
        if st.sidebar.button('Train'):
            s1 = ClassificationExperiment()
            s1.setup(data=data, target=target)
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
            s2.setup(data=data, target=target)
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
