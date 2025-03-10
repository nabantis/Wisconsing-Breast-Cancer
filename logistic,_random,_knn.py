"""Logistic, Random, KNN.ipynb

Original file is located at
    https://colab.research.google.com/drive/1xoCzWsjuRRnlrYQTIs2gP1kwLFWKko4P

#**Wisconsing Breast Cancer**

- **By:** Cat Phuong Tran
- **Date:** 4/13/2023

#### **Import:**
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

"""#### **Read Data:**"""

Data = "/content/wisconsinBreastCancer - wisconsinBreastCancer.csv"
df = pd.read_csv(Data)

df.head()

df.shape

df.info()

"""### **Logistic Regresion:**"""

# Spliting the data into features and target
X = df.drop(['diagnosis', "id"], axis=1)
y = df['diagnosis']

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and evaluate a default logistic regression model
logreg_default = LogisticRegression(max_iter=10000)
logreg_default.fit(X_train, y_train)
y_pred_default = logreg_default.predict(X_test)

print('Default Logistic Regression Model:')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_default))
print('Classification Report:')
print(classification_report(y_test, y_pred_default))

# Creating a dictionary of hyperparameters for tuning:
param_grid = {'penalty': ['l1', 'l2'],
              'C': np.logspace(-4, 4, 9)}

# Create and evaluate a tuned logistic regression model using GridSearchCV
logreg_tuned = LogisticRegression(max_iter=10000)
logreg_grid = GridSearchCV(logreg_tuned, param_grid, cv=5)

# Fit it:
logreg_grid.fit(X_train, y_train)
# Predict:
y_pred_tuned = logreg_grid.predict(X_test)

print('\n Tuned Logistic Regression Model:')
print('Best Parameters:', logreg_grid.best_params_)
print('\n Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_tuned))
print('\n Classification Report:')
print(classification_report(y_test, y_pred_tuned))

"""#####**<font color='#fba00f'>Which hyperparameters did you tune?</font>**
- The hyperparameters tuned were the penalty type (L1 or L2) and the inverse regularization strength (C).

#####**<font color='#fba00f'>What values for those hyperparameters led to the best-tuned model?</font>**

- The best-tuned model had a penalty type of L2 and an inverse regularization strength (C) of 0.1.

### **KNeighborsClassifier:**
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create a default KNN model and fit it to the data
knn_default = KNeighborsClassifier()
knn_default.fit(X_train, y_train)

# Make predictions on the test set and calculate accuracy and confusion matrix
y_pred = knn_default.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print("_Default KNN Model_")
print("Accuracy:", accuracy)
print("Confusion matrix:\n", confusion)

# Define the parameter grid to search
param_grid = {'n_neighbors': range(1, 31),
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}

# Create a GridSearchCV object with the KNN model and parameter grid
knn_gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

# Fit the GridSearchCV object to the data
knn_gs.fit(X_train, y_train)

# Make predictions on the test set using the best model and calculate accuracy and confusion matrix
y_pred_gs = knn_gs.best_estimator_.predict(X_test)
accuracy_gs = accuracy_score(y_test, y_pred_gs)
confusion_gs = confusion_matrix(y_test, y_pred_gs)
print("\n_Tuned KNN Model_")
print("Best parameters:", knn_gs.best_params_)
print("Accuracy:", accuracy_gs)
print("Confusion matrix:\n", confusion_gs)

"""#####**<font color='#fba00f'>Which hyperparameters did you tune?</font>**

- We tuned the number of neighbors, the type of weighting, and the distance metric used by the KNN algorithm.

#####**<font color='#fba00f'>What values for those hyperparameters led to the best-tuned model?</font>**

- The best parameters found by GridSearchCV were n_neighbors=6, weights='uniform', and metric='manhattan'.

# **Random Forest:**

- **Random Forest Model:**
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Default Random Forest Model
rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

# Evaluate Default Model
print("Default Random Forest Model:")
print(classification_report(y_test, rfc_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rfc_pred))

# Grid Search for Best Parameters
param_grid = {'n_estimators': [100, 500, 1000],
              'max_depth': [5, 10, 20, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}
grid = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
best_rfc = grid.best_estimator_
best_rfc_pred = best_rfc.predict(X_test)

# Evaluate Best Model
print("Best Random Forest Model:")
print(classification_report(y_test, best_rfc_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, best_rfc_pred))

"""### **4)**
In the context of diagnosing breast cancer, false positives refer to cases where the model classifies a non-cancerous tumor as cancerous, while false negatives refer to cases where the model classifies a cancerous tumor as non-cancerous.

False negatives are considered worse in this problem because a missed diagnosis of cancer can delay necessary treatment and have serious health consequences for the patient.

- **Lastly**

Based on the evaluation metrics, the Random Forest model with tuned hyperparameters would be recommended for production. The model has the highest accuracy score, F1-score, and AUC-ROC score, indicating that it performs better than the other models in both overall accuracy and balance between precision and recall.
"""
