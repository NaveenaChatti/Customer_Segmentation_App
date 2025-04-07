import streamlit as st
import numpy as np

import pickle

kmeans_new=pickle.load(open('kmeans_new.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))


# Define a Clustering function
def clustering(age,purchase):
    new_record=np.array([[age,purchase]])
    scaled_record=scaler.transform(new_record)
    predicted_cluster=kmeans_new.predict(scaled_record)


    if predicted_cluster[0]==0:
       return "Emerging Shoppers"
    elif predicted_cluster[0]==1:
       return "Prime Spenders"
    else:
        return "Golden Year Economists"
    
      

# Streamlit App Building

st.markdown("# Welcome to Walmart Customer Categorization Model App")
st.markdown("## Enter Customer Information Below")
st.markdown("Use **Age** and **Purchase amount** to predict the customer category.")

col1,col2=st.columns(2)

with col1:
    st.subheader('Customer Age')
    age=st.number_input('Age',min_value=17,max_value=75, value=17)
with col2:
    st.subheader('Total Purchase Amount')
    purchase=st.number_input('Purchase',min_value=0.0,max_value=9999999.0, value=0.0)

if st.button('Customer Category'):
    cluster_label=clustering(age,purchase)
    st.success(f'The customer belongs to "{cluster_label}" category')



