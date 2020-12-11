"""
Author : Akhil Nair
Date : 9/20/2020
Version : 1.0
"""
from operator import add
import streamlit as st
import pandas as pd
import engine
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

st.set_page_config(layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style.css")

test_accuracy = 0
train_accuracy = 0
macro_f1 = 0

# Streamlit configurations
add_selectbox = st.sidebar.radio("",("Poster","Data & EDA","Logistic Regression","Decision Tree","Naive Bayes","SVM"))
st.sidebar.write('Created by :')
st.sidebar.write('Akshaya Mohan')
st.sidebar.write('Lavnish Talreja')
st.sidebar.write('Akhil Nair')
config = {}
if add_selectbox == "Poster":
    st.markdown("<h1 style='text-align: center; color: black;'>Incident Management System</h1>", unsafe_allow_html=True)
    image = Image.open('Data/poster_updated.jpg')
    st.image(image, caption='Incident Management',use_column_width=True)

if add_selectbox == "Data & EDA":

    st.markdown("<h1 style='text-align: center; color: black;'>Data</h1>", unsafe_allow_html=True)
    

    image = Image.open('Data/corr.jpg')
    st.markdown("<h1 style='text-align: center; color: black;'>Correlation Matrix</h1>", unsafe_allow_html=True)
    st.image(image,use_column_width=True)
    st.markdown("<h3 style='text-align: center; color: black;'>Pearson correlation was used to calculate the feature importance. The most important features were SLA and IMPACT which had highest coeefecients with respect to the target feature i.e Priority</h3>", unsafe_allow_html=True)
    

    image = Image.open('Data/distribution.png')
    st.markdown("<h1 style='text-align: center; color: black;'>Distribution of Incident State</h1>", unsafe_allow_html=True)
    st.image(image, caption='Distribution of Incident State',use_column_width=True)

    image = Image.open('Data/code.png')
    st.markdown("<h1 style='text-align: center; color: black;'>Distribution of Closed code</h1>", unsafe_allow_html=True)
    st.image(image, caption='Distribution of Closed code',use_column_width=True)
        
if add_selectbox == "Logistic Regression":
    st.markdown("<h1 style='text-align: center; color: black;'>Logistic Regression</h1>", unsafe_allow_html=True)
    col1, col2 = st.beta_columns((1, 2))
    with col1:
        st.markdown("<h3 style='text-align: center; color: black; font-weight: bold;'>Hyperparameters</h1>", unsafe_allow_html=True)
    c_value = col1.selectbox('C Value',(0.1, 1, 100))
    penalty = col1.selectbox('Penalty',('l1','l2'))
    solver = col1.selectbox('Solver',('liblinear', 'newton-cg','lbfgs'))
    hyperparameters = {'c' : c_value,'penalty' : penalty,'solver' : solver}

    config['model'] = 'lr'
    config['hyperparameters'] = hyperparameters
    with col1:
        run = st.button('Run Model')
    if run:
        train_accuracy,test_accuracy,macro_f1,priority_df = engine.run(config)
        image = Image.open('Data/lr_roc.png')
        col2.image(image, caption='Receiver Operating Characteristic -  LR',use_column_width=True)
        st.markdown("<h1 style='text-align: center; color: black;'>High Priority Incidents</h1>", unsafe_allow_html=True)
        st.write(priority_df)
    with col1:
        st.markdown("<h3 style='text-align: center; color: black; font-weight: bold;'>Evaluation Metrics</h1>", unsafe_allow_html=True)
        st.write('Training Accuracy: ', train_accuracy)
        st.write('Testing Accuracy : ',test_accuracy)
        st.write('Macro F1 score : ',macro_f1)


elif add_selectbox == 'Decision Tree':
    st.markdown("<h1 style='text-align: center; color: black;'>Decision Tree</h1>", unsafe_allow_html=True)
    col1, col2 = st.beta_columns((1, 2))
    with col1:
        st.markdown("<h3 style='text-align: center; color: black; font-weight: bold;'>Hyperparameters</h1>", unsafe_allow_html=True)
    criterion = col1.selectbox('Criterion',('gini','entropy'))
    min_samples_split = col1.selectbox('Min sample split',(10,20))
    max_depth = col1.selectbox('Max depth',(2,3))
    max_leaf_nodes = col1.selectbox('Max leaf nodes',(5,10,30))
    hyperparameters = {'criterion' : criterion, 'min_samples_split':min_samples_split,'max_depth':max_depth,'max_leaf_nodes':max_leaf_nodes}
    config['model'] = 'dt'
    config['hyperparameters'] = hyperparameters
    with col1:
        run = st.button('Run Model')
    if run:
        train_accuracy,test_accuracy,macro_f1,priority_df = engine.run(config)
        image = Image.open('Data/decistion_tree.png')
        col2.image(image, caption='Decision Tree',use_column_width=True)
        st.markdown("<h1 style='text-align: center; color: black;'>High Priority Incidents</h1>", unsafe_allow_html=True)
        st.write(priority_df)
    with col1:
        st.markdown("<h3 style='text-align: center; color: black; font-weight: bold;'>Evaluation Metrics</h1>", unsafe_allow_html=True)
        st.write('Training Accuracy: ', train_accuracy)
        st.write('Testing Accuracy : ',test_accuracy)
        st.write('Macro F1 score : ',macro_f1)
    

elif add_selectbox == 'Naive Bayes':
    accuracy = 0
    st.markdown("<h1 style='text-align: center; color: black;'>Naive Bayes</h1>", unsafe_allow_html=True)
    col1, col2 = st.beta_columns((1, 2))
    with col1:
        st.markdown("<h3 style='text-align: center; color: black; font-weight: bold;'>Hyperparameters</h1>", unsafe_allow_html=True)
    alpha = col1.selectbox('alpha',(0.5, 1, 1.5))
    fit_prior = col1.selectbox('Fit prior',(True, False))
    hyperparameters = {'alpha' : alpha,'fit_prior':fit_prior}
    config['model'] = 'nbc'
    config['hyperparameters'] = hyperparameters
    with col1:
        run = st.button('Run Model')

    if run:
        train_accuracy,test_accuracy,macro_f1,priority_df = engine.run(config)
        image = Image.open('Data/nbc_roc.png')
        col2.image(image, caption='Receiver Operating Characteristic - NBC',use_column_width=True)
        st.markdown("<h1 style='text-align: center; color: black;'>High Priority Incidents</h1>", unsafe_allow_html=True)
        st.write(priority_df)
    with col1:
        st.markdown("<h3 style='text-align: center; color: black; font-weight: bold;'>Evaluation Metrics</h1>", unsafe_allow_html=True)
        st.write('Train Accuracy: ', train_accuracy)
        st.write('Test Accuracy: ', test_accuracy)
        st.write('Macro F1 score : ',macro_f1)

    engine.run(config)
    
elif add_selectbox == 'SVM':
    accuracy = 0
    macro_f1 = 0
    st.markdown("<h1 style='text-align: center; color: black;'>SVM</h1>", unsafe_allow_html=True)
    col1, col2 = st.beta_columns((1, 2))
    with col1:
        st.markdown("<h3 style='text-align: center; color: black; font-weight: bold;'>Hyperparameters</h1>", unsafe_allow_html=True)
    c_value = col1.selectbox('C',(0.1, 1, 100))
    gamma_value = col1.selectbox('Gamma',(1, 0.1, 0.01))
    kernel = col1.selectbox('Kernel',('linear', 'rbf'))
    hyperparameters = {'c' : c_value,'g' : gamma_value,'kernel' : kernel}
    with col1:
        run = st.button('Run Model')

    config['model'] = 'svm'
    config['hyperparameters'] = hyperparameters

    if run:
        train_accuracy,test_accuracy,macro_f1,priority_df = engine.run(config)
        image = Image.open('Data/svm_roc.png')
        col2.image(image, caption='Receiver Operating Characteristic -  SVM',use_column_width=True)
        st.markdown("<h1 style='text-align: center; color: black;'>High Priority Incidents</h1>", unsafe_allow_html=True)
        st.write(priority_df)
    with col1:
        st.markdown("<h3 style='text-align: center; color: black; font-weight: bold;'>Evaluation Metrics</h1>", unsafe_allow_html=True)
        st.write('Train Accuracy: ', train_accuracy)
        st.write('Test Accuracy: ', test_accuracy)
        st.write('Macro F1 score : ',macro_f1)