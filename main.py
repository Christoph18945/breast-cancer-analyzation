#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""app.py

Main program and entrypoint for breast cancer analyzation. Goal is
to predict th probability of predicitng if breast cancer is begnin or malignant.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def main():
    """Main program predicts the probability of classifying
    breast cancer as being begnin or malignant."""
    
    # Read in data from csv file.
    df: pd.DataFrame = pd.read_csv(r"data/cancer.csv")

    # Remove coumn with label id.
    # Along the first axis.
    df = df.drop("id", axis = 1)

    # Predict the value of the column "diagonsis" by other values in the table
    # 
    # What column should be used to predict the value:
    # X = df[["age", "interest"]].values
    # NOTE: The Column "success" shoul dnot be used to predict the value

    # Remove column labeled "diagnosis" along
    # the first axis. Other columns remain. Note
    # that a copy of the original Dataframe is 
    # created and it does not replace the original.
    X = df.drop("diagnosis", axis = 1).values

    # Extract the underlying NumPy from the
    # "diagnosis" column. The diagnosis column
    # is used to perform the prediction.
    y = df["diagnosis"].values

    # Split data into training and test data:
    # random_state = 0 sets a random seed toensure reproducability
    # test_size = 0.25 speicfies that 25% of the data will be used for testing and
    # the remaining 75% will be used for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

    # fit scaliung parameters for the
    # training data set
    scaler = StandardScaler()
    scaler.fit(X_train)

    # transform training and testing data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # initialize the logistic regression model
    model = LogisticRegression()
    # fit the model for training data
    model.fit(X_train, y_train)

    # Calculate the accuracy of the model on the provided
    # test data and provided labels. the output value
    # represent the ratio of correctly prediciting the
    # classification of breast cancer being begnin or
    # malignant.
    print(model.score(X_test, y_test))

    return None

if __name__ == "__main__":
    main()
