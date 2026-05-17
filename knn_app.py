#create a streamlit application for knn classifier reference is knn.ipynb
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
st.title("K-Nearest Neighbors (KNN) Classifier")
st.write("This application allows you to train a K-Nearest Neighbors (KNN) classifier on the Iris dataset.")
# Select features and target variable
features = st.multiselect("Select features for training:", df.columns[:-1].tolist(), default=df.columns[:-1].tolist())
target = st.selectbox("Select target variable:", df.columns[-1])
# Train the KNN classifier
if st.button("Train KNN Classifier"):
    X = df[features]
    y = df[target]
    model = neighbors.KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    st.success("KNN Classifier trained successfully!")
    # Display the accuracy of the trained model
    accuracy = model.score(X, y)
    st.write(f"Accuracy of the trained KNN model: {accuracy:.2f}")
    # Display the predictions of the trained model
    predictions = model.predict(X)
    st.write("Predictions of the trained KNN model:")
    st.write(predictions)
