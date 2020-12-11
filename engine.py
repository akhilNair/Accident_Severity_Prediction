#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
import graphviz 
from sklearn import tree
from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import random


data = pd.read_csv('Data/reopened.csv')
data = data.iloc[:,1:]
data.head()

X = data.loc[:,data.columns != 'priority']
y = data['priority']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def _run_lr(hyperparameters):
    c = hyperparameters['c']
    penalty = hyperparameters['penalty']
    solver = hyperparameters['solver']
    clf = LogisticRegression(penalty = penalty, C = c,solver = solver)
    clf.fit(X_train,y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    y_score = clf.predict_proba(X_test)[:,1]
    print('score : ',y_score)
    fpr, tpr, threshold = roc_curve(y_test, y_score)

    y_pred_final = clf.predict(X)
    y_pred_index = [i for i,y in enumerate(y_pred_final) if y ==1]

    priority_df = data.iloc[y_pred_index,:]

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - Logistic Regression')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('Data/lr_roc.png')

    roc = roc_auc_score(y_test, y_score)
    target_names = ['Low priority','High Priority']
    report = classification_report(y_test, y_pred_test, target_names=target_names,output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    return train_accuracy,test_accuracy,macro_f1,priority_df


def _run_dt(hyperparameters):
    criterion = hyperparameters['criterion']
    min_samples_split = hyperparameters['min_samples_split']
    max_depth = hyperparameters['max_depth']
    max_leaf_nodes = hyperparameters['max_leaf_nodes']
    dt_tuned = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, max_leaf_nodes =max_leaf_nodes,min_samples_split = min_samples_split)
    dt = dt_tuned.fit(X_train, y_train)

    y_pred_tuned_train = dt_tuned.predict(X_train)
    y_pred_tuned_test = dt_tuned.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_tuned_train)
    test_accuracy = accuracy_score(y_test, y_pred_tuned_test)

    target_names = ['Low priority','HIgh priority']
    report = classification_report(y_test, y_pred_tuned_test, target_names=target_names,output_dict=True)
    macro_f1 = report['macro avg']['f1-score']

    fig = plt.figure(figsize=(100,55))

    feature_names = X.columns

    _ = tree.plot_tree(dt, feature_names = feature_names,
                    class_names=['Low Priority','High Priority'],
                    filled=True) 
    fig.savefig("Data/decistion_tree.png")
    
    y_score = dt_tuned.predict_proba(X_test)[:,1]
    print('score : ',y_score)
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_score))

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - DecisionTree')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('Data/dt_roc.png')

    roc = roc_auc_score(y_test, y_score)

    y_pred_final = dt_tuned.predict(X)
    print(y_pred_final)
    print('type : ',type(y_pred_final))
    y_pred_index = [i for i,y in enumerate(y_pred_final) if y ==1]

    priority_df = data.iloc[y_pred_index]

    return train_accuracy,test_accuracy,macro_f1,priority_df

def _run_svm(hyperparameters):
    c = hyperparameters['c']
    g = hyperparameters['g']
    kernel = hyperparameters['kernel']
    clf = SVC(C = c,gamma = g, kernel = kernel)
    clf.fit(X_train, y_train) 

    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    train_accuracy = accuracy_score(y_pred_train,y_train)
    test_accuracy = accuracy_score(y_pred_test,y_test)

    target_names = ['Low Priority','High Priority']
    report = classification_report(y_test, y_pred_test, target_names=target_names,output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    fpr, tpr, threshold = roc_curve(y_test, y_pred_test)

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - SVM')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('Data/svm_roc.png')

    y_pred_final = clf.predict(X)
    y_pred_index = [i for i,y in enumerate(y_pred_final) if y ==1]

    priority_df = data.iloc[y_pred_index]
    return train_accuracy,test_accuracy,macro_f1,priority_df

def _run_nbc(hyperparameters):
    alpha = hyperparameters['alpha']
    fit_prior = hyperparameters['fit_prior']
    clf = MultinomialNB(alpha = alpha,fit_prior= fit_prior)

    clf.fit(X_train, y_train) 

    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    test_accuracy = accuracy_score(y_pred_test,y_test)
    train_accuracy = accuracy_score(y_pred_train,y_train)

    fpr, tpr, threshold = roc_curve(y_test, y_pred_test)

    target_names = ['Low Priority','High Priority']
    report = classification_report(y_test, y_pred_test, target_names=target_names,output_dict=True)
    macro_f1 = report['macro avg']['f1-score']

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - NBC')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('Data/nbc_roc.png')

    y_pred_final = clf.predict(X)
    print(y_pred_final)
    print('type : ',type(y_pred_final))
    y_pred_index = [i for i,y in enumerate(y_pred_final) if y ==1]

    priority_df = data.iloc[y_pred_index]
    return train_accuracy,test_accuracy,macro_f1, priority_df

def run(config):
    model = config['model']
    hyperparameters = config['hyperparameters']
    
    if model == 'lr':
        return _run_lr(hyperparameters)
    elif model == 'dt':
        return _run_dt(hyperparameters)
    elif model == 'svm':
        return _run_svm(hyperparameters)
    elif model == 'nbc':
        return _run_nbc(hyperparameters)
