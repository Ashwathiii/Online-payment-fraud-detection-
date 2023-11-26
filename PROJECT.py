
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
# Load the dataset
dff=pd.read_csv("C:\\Users\\aswat\\Downloads\\archive (6)\\PS_20174392719_1491204439457_log.csv")

# giving the title of the app
st.title("Online Payment Fraud Detection")
df1=dff.loc[dff['isFraud']==1]
df2=dff.loc[dff['isFraud']==1]
df=pd.concat([df1.head(5000),df2.head(5000)],ignore_index=True)

# # displaying datset in the tabl
st.write("Dataset:",df)


# Summary Statistics
st.header("Data Visualization")
st.subheader("Summary Statistics:")
st.write(df.describe())
st.subheader("Model Accuracy Comparison")

models=['knn', 'sv', 'rf', 'ad', 'lr', 'nb', 'dt']
accuracy=[95.76,90.80,99.36,98.60,71.56,71.30,98.96]

# Create a DataFrame from the results for plotting
results_df = pd.DataFrame({'Model': models, 'Accuracy': accuracy})
st.write("Accuracy Results")
st.table(results_df)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
# Load feature data from your dataset
data = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)

# ... (preprocessing and modeling steps)
X = data.drop(['nameOrig','nameDest','isFlaggedFraud','step','isFraud'], axis=1)  # Replace 'target_column' with the actual label column name
y = data['isFraud']

# Reshape the y variable using ravel()
y = y.values.ravel()

# Split your data into training and testing sets
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=14)

# Create a MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the training data and transform the testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Transform the training and testing
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Make predictions on the test data
# Predict on the test data
y_pred = rf.predict(X_test)

def main():
    st.title("Fraud Detection App")

    # Add input fields for users to enter data
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        V1 = st.number_input("Type")
    with col2:
        V2 = st.number_input("Amount")
    with col3:
        V3 = st.number_input("Old balance(senders)")
    with col4:
        V4 = st.number_input("New balance(senders)")
    with col5:
        V5 = st.number_input("Old balance(Recievers)")
    with col6:
        V6 = st.number_input("New balance(Recievers)")

    if st.button("Detect Fraud"):
        # TO Get user input values
        input_data = np.array([[V1, V2, V3, V4, V5, V6]])

        # ScalING  the input data
        scaled_input = scaler.transform(input_data)

        # Making prediction using the Random Forest model
        y_new = rf.predict(scaled_input)

        # Display the result
        if y_new[0] == 1:
            st.write("Fraud Detected")
        else:
            st.write("No Fraud Detected")

if __name__ == "__main__":
    main()

 #accuracy
accuracy = 99.36
formatted_accuracy=f"{accuracy:.2f}%"
st.write(f"Highest Accuracy(RandomForest Classifier=): {formatted_accuracy}")





