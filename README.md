# Walmart Customer Segmentation App

This project helps to categorize Walmart customers based on their **age** and **purchase** habits using **KMeans clustering**. A **Streamlit app** allows users to input customer data (age and purchase amount) and predict which category the customer belongs to.

## Key Actions Taken:

- **Data Preprocessing**:
  - Loaded the Walmart customer dataset.
  - Selected **age** and **purchase** features for clustering.
  - Grouped data by age and summed the total purchase amount for each customer.

- **Data Scaling**:
  - Applied **StandardScaler** to normalize the data (age and purchase values) for clustering.

- **KMeans Clustering**:
  - Used the **Elbow method** to find the optimal number of clusters (k=6).
  - Ran **KMeans clustering** to segment customers into 3 groups based on their age and purchase behavior:
    - **Emerging Shoppers**
    - **Prime Spenders**
    - **Golden Year Economists**
  - Evaluated the clustering model using the **Silhouette score** for cluster validity with highest score of 0.48 for 3 clusters.
  - Model executed again with 3 clusters.
 
- **Streamlit App Development**:
  - Built a web app using **Streamlit** to predict customer categories based on input age and purchase amount.
  - Users enter **age** and **purchase** amount to receive a prediction for the customer's category.
  
- **Model Deployment**:
  - Saved the **KMeans model** and **scaler** using **Pickle** for later use in the app.
  - Loaded the saved model in the Streamlit app to predict customer categories without retraining.

## Tech Used

- **Python** for everything
- **Pandas & NumPy** for data handling
- **Scikit-learn** for clustering and scaling
- **Streamlit** to build the app
- **Pickle** to save the model

The app takes **age** and **purchase amount** as input and predicts which category the customer belongs to. Here's how it works:

1. You enter **age** and **purchase amount**.
2. The model predicts which of the three customer categories the person belongs to.
3. The result shows up on the screen.

