# Walmart Customer Segmentation

This project segments Walmart customers into categories based on their demographic and spending data using clustering algorithms. A **Streamlit app** is also created to predict customer categories based on age and spending.

## Table of Contents
- [Overview](#overview)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Clustering Process](#clustering-process)
- [Streamlit App](#streamlit-app)
- [Model Deployment](#model-deployment)
- [Getting Started](#getting-started)
- [Conclusion](#conclusion)

## Overview

This project segments customers into clusters like **Emerging Shoppers**, **Prime Spenders**, and **Golden Year Economists** based on their age and purchase behavior using **KMeans clustering**. The Streamlit app allows users to input customer data and predict the customer category.

## Technologies

- Python
- Pandas, NumPy
- Scikit-learn (KMeans, Scaling)
- Streamlit (for app)
- Pickle (for saving models)

## Dataset

The dataset contains:
- `User_ID`: Customer ID
- `Age`: Customer's age
- `Purchase`: Total spending

## Clustering Process

- **Preprocessing**: Grouped by `User_ID` and `Age`, then scaled the data using `StandardScaler`.
- **KMeans**: Used KMeans to find 3 customer segments.
- **Silhouette Score**: Used for cluster validation.

## Streamlit App

The app predicts customer categories based on input age and spending.

### App Code:

```python
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
kmeans_new = pickle.load(open('kmeans_new.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def clustering(age, purchase):
    new_record = np.array([[age, purchase]])
    scaled_record = scaler.transform(new_record)
    predicted_cluster = kmeans_new.predict(scaled_record)
    return ["Emerging Shoppers", "Prime Spenders", "Golden Year Economists"][predicted_cluster[0]]

st.markdown("# Walmart Customer Categorization App")
age = st.number_input('Customer Age', 17, 75)
purchase = st.number_input('Purchase Amount', 0.0, 9999999.0)

if st.button('Predict Category'):
    cluster_label = clustering(age, purchase)
    st.success(f'The customer belongs to "{cluster_label}" category')
