# 🍽️ Restaurant Rating Prediction 

# 📌 Introduction

Restaurant ratings play an important role in customer decision-making and business growth.  
Factors such as cuisines, pricing, customer votes, online delivery services, and table booking facilities can influence restaurant ratings significantly.

This project uses **Machine Learning** to predict restaurant ratings based on restaurant-related features.  
A **Streamlit web application** is also developed to provide an interactive and user-friendly interface where users can enter restaurant information and instantly get predicted ratings.

The project demonstrates the complete Machine Learning workflow from data preprocessing to deployment-ready web application development.

---

# 🎯 Project Objective

The main objectives of this project are:

- To build a Machine Learning model capable of predicting restaurant ratings.
- To understand real-world Machine Learning workflow.
- To perform data preprocessing and feature engineering.
- To create an interactive Streamlit web application.
- To save and reuse trained Machine Learning models.
- To manage and share the project using GitHub.

---

# 🧠 Problem Statement

Restaurant businesses receive large amounts of customer reviews and ratings every day.  
Analyzing and predicting ratings manually becomes difficult as the amount of data increases.

Using Machine Learning, this project aims to:

- Analyze restaurant-related features
- Learn patterns from historical data
- Predict future restaurant ratings automatically
- Improve decision-making systems

This project solves the problem by building a regression-based Machine Learning prediction system.

---

# 📂 Dataset Information

The dataset contains restaurant-related information such as:

| Column Name | Description |
|---|---|
| Restaurant Name | Name of restaurant |
| City | City location |
| Cuisines | Types of cuisines served |
| Average Cost for two | Estimated cost |
| Currency | Currency type |
| Has Table booking | Booking availability |
| Has Online delivery | Online delivery service |
| Price range | Restaurant pricing category |
| Votes | Number of customer votes |
| Aggregate rating | Final restaurant rating |

---

# ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Streamlit | Web application framework |
| Joblib | Model saving/loading |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Git & GitHub | Version control and project hosting |

---

# 🔄 Machine Learning Workflow

The project follows a complete Machine Learning pipeline.

---

## 1️⃣ Importing Libraries

Necessary Python libraries are imported for:

- Data manipulation
- Visualization
- Model training
- Evaluation
- Saving models

Example:

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
