# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:25:47 2023

@author: kdeep
"""

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pf
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, ClassificationExperiment
from pycaret.regression import setup, compare_models, pull, save_model, RegressionExperiment 


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
    st.write('Upload your Dataset')
    upload = st.file_uploader('Upload here')
    if upload:
        data = pd.read_csv(upload, index_col=None)
        data.to_csv("sourcev.csv", index=None)
        st.dataframe(data)
        st.success('Dataset uploaded successfully!')
    else:
        if os.path.exists("sourcev.csv"):
            data = pd.read_csv("sourcev.csv", index_col=None)
            st.dataframe(data)
        else:
            st.warning('No dataset available. Please upload a dataset.')

        
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
            s2.save_model(best_model2,"Machine learning Model")

if choose =="Download":
    with open("Machine Learning model.pkl",'rb') as f:
        st.caption("Download your model from here:")
        st.download_button("Download the file",f,"Machine Learning model.pkl")
