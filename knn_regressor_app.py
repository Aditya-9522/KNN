#create a streamlit application for knn regressor reference is knn.ipynb
import streamlit as st
import pandas as pd
from sklearn import neighbors
from sklearn.datasets import load_iris
# Load the iris dataset
iris = load_iris()
# Create a DataFrame from the iris dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add the target variable to the DataFrame
df['target'] = iris.target
# Create a Streamlit application
st.title("K-Nearest Neighbors (KNN) Regressor")
st.write("This application allows you to train a K-Nearest Neighbors (KNN) regressor on the Iris dataset.")
# Select features and target variable
features = st.multiselect("Select features for training:", df.columns[:-1].tolist(), default=df.columns[:-1].tolist())
target = st.selectbox("Select target variable:", df.columns[-1])
# Train the KNN regressor
if st.button("Train KNN Regressor"):
    X = df[features]
    y = df[target]
    model = neighbors.KNeighborsRegressor(n_neighbors=3)
    model.fit(X, y)
    st.success("KNN Regressor trained successfully!")
    # Display the predictions of the trained model
    predictions = model.predict(X)
    st.write("Predictions of the trained KNN regressor:")
    st.write(predictions)
