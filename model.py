#!/usr/bin/env python
# coding: utf-8

# # Restaurant Rating Prediction using ML

# In[1]:


# Import libraries
import pandas as pd
import numpy as np


# In[2]:


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


# In[4]:


# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[10]:


# Load Dataset
df = pd.read_csv("Dataset .csv")


# In[11]:


# Display first 5 rows
print(df.head())


# In[12]:


# Handle Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())


# In[15]:


# Fill missing categorical values
df["Cuisines"] = df["Cuisines"].fillna("Unknown")


# In[16]:


# Encode Categorical Variables
# Convert categorical columns into numeric
# Create an empty dictionary to store encoders
label_encoders = {} 

categorical_columns = [
    "City", "Cuisines", "Currency", "Has Table booking", 
    "Has Online delivery", "Is delivering now", 
    "Switch to order menu", "Rating color", "Rating text"
]

for col in categorical_columns:
    # Create a NEW encoder for each column
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    # Store it in the dictionary using the column name as the key
    label_encoders[col] = le


# In[17]:


# Select Features and Target
# Features (X)
X = df.drop(columns=[
    "Aggregate rating",
    "Restaurant Name",
    "Address",
    "Locality",
    "Locality Verbose"
])
# Target variable (y)
y = df["Aggregate rating"]


# In[18]:


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# In[19]:


# Linear Regression Model
lr_model = LinearRegression()


# In[20]:


# Train model
lr_model.fit(X_train, y_train)


# In[21]:


# Predictions
lr_predictions = lr_model.predict(X_test)


# In[22]:


# Evaluation
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print("\n---- Linear Regression Results ----")
print("Mean Squared Error:", lr_mse)
print("R2 Score:", lr_r2)


# In[24]:


# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)


# In[25]:


# Train model
dt_model.fit(X_train, y_train)


# In[26]:


# Predictions
dt_predictions = dt_model.predict(X_test)


# In[27]:


# Evaluation
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)

print("\n---- Decision Tree Results ----")
print("Mean Squared Error:", dt_mse)
print("R2 Score:", dt_r2)


# In[28]:


# Feature Importance (Decision Tree)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt_model.feature_importances_
})
importance = importance.sort_values(by="Importance", ascending=False)
print("\n---- Feature Importance ----")
print(importance)


# In[29]:


# Plot Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(x="Importance",y="Feature",data=importance)
plt.title("Feature Importance for Restaurant Ratings")
plt.show()


# In[30]:


# Interpretation
print("\nModel Interpretation:")
print("Features with higher importance have greater influence on restaurant ratings.")
print("Common influential features may include:")
print("- Votes")
print("- Price range")
print("- Online delivery")
print("- Table booking")
print("- Cuisine type")


# In[ ]:


# SAVE FILES

import joblib

joblib.dump(dt_model, "model.pkl")

joblib.dump(label_encoders, "label_encoders.pkl")

joblib.dump(X.columns.tolist(), "features.pkl")

print("Files Saved Successfully")


