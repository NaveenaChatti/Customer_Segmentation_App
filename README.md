# Walmart Customer Segmentation

This project helps to categorize Walmart customers based on their **age** and **purchase** habits using **KMeans clustering**. A **Streamlit app** allows users to input customer data (age and purchase amount) and predict which category the customer belongs to.

## What’s Inside
- **Overview**: How we segment customers and predict categories
- **Tech Used**: Tools and libraries used in the project
- **Dataset**: Information about the data
- **Clustering**: How the customer groups were created
- **Streamlit App**: How the app works
- **Model Deployment**: Saving and using the model for predictions

## Tech Used

- **Python** for everything
- **Pandas & NumPy** for data handling
- **Scikit-learn** for clustering and scaling
- **Streamlit** to build the app
- **Pickle** to save the model

## Dataset

The dataset contains data on **550,068 customers**, including information like:

- Age
- Purchase amount
- Other features, but only **age** and **purchase** were used for clustering

## Clustering Process

1. **Preprocessing**: Grouped customers by age and summed their purchases.
2. **Standardizing**: Scaled the data to make it ready for clustering.
3. **Clustering**: Used **KMeans** to group customers into three categories based on their spending behavior:
   - **Emerging Shoppers**
   - **Prime Spenders**
   - **Golden Year Economists**

## Streamlit App

The app takes **age** and **purchase amount** as input and predicts which category the customer belongs to. Here's how it works:

1. You enter **age** and **purchase amount**.
2. The model predicts which of the three customer categories the person belongs to.
3. The result shows up on the screen.

