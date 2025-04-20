# Heart Disease Prediction Using Machine Learning

This project focuses on predicting heart disease using machine learning techniques, leveraging various classification algorithms to compare their effectiveness. The aim is to build a robust model that can predict the likelihood of heart disease in individuals based on features like age, sex, cholesterol levels, blood pressure, and more.

### Problem Statement
Heart disease is one of the leading causes of death globally. Early detection can help save lives, and this project aims to develop a machine learning model that can predict whether a person has heart disease based on their medical data. By comparing multiple algorithms, we aim to identify the most accurate model for this task.

### Key Features of the Dataset
The dataset used in this project is the **Heart Disease UCI dataset**, which contains various health features of individuals. Some of the key features include:
- **Age**: Age of the individual.
- **Sex**: Gender of the individual.
- **Chest Pain Type**: Type of chest pain experienced.
- **Resting Blood Pressure**: Blood pressure levels.
- **Cholesterol**: Serum cholesterol in mg/dl.
- **Fasting Blood Sugar**: Fasting blood sugar levels.
- **Max Heart Rate**: Maximum heart rate achieved during exercise.
- **Exercise-Induced Angina**: Whether the individual has exercise-induced angina.
- **Oldpeak**: Depression induced by exercise relative to rest.
- **Slope of Peak Exercise ST Segment**: Slope of the ST segment during exercise.

The target variable is whether the person has heart disease or not, represented by `0` (no) or `1` (yes).

### Models Used
In this project, I implemented and compared several machine learning classification algorithms:
- **Logistic Regression**: A linear model used for binary classification.
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
- **Support Vector Machine (SVM)**: A powerful classification algorithm that uses hyperplanes.
- **K-Nearest Neighbors (KNN)**: A simple yet effective algorithm based on proximity.
- **Decision Tree**: A tree-based model that splits data based on feature values.
- **Random Forest**: An ensemble method that uses multiple decision trees to improve accuracy.
- **XGBoost**: An optimized gradient boosting algorithm.
- **Neural Network (MLP)**: A multi-layer perceptron for capturing non-linear patterns.

### Key Findings
- The **Random Forest** model gave the best performance with an accuracy of **90.16%**.
- **Logistic Regression** and **Naive Bayes** also showed strong performance with an accuracy of **85.25%**.
- **XGBoost** and **Neural Networks** achieved moderate accuracy but were less effective compared to Random Forest.

| **Model**              | **Accuracy (%)** |
|------------------------|------------------|
| Logistic Regression    | 85.25            |
| Naive Bayes            | 85.25            |
| Support Vector Machine | 81.97            |
| K-Nearest Neighbors    | 67.21            |
| Decision Tree          | 81.97            |
| **Random Forest**      | **90.16**        |
| XGBoost                | 81.97            |
| Neural Network (MLP)   | 83.61            |

### Visualizations
To better understand the model performance and data, several visualizations were created, including:
- **Accuracy Comparison Bar Chart**: A bar chart showing the comparison of model accuracy.
- **Confusion Matrix**: A heatmap displaying the true vs. predicted values for each model.
- **Feature Importance**: A bar plot showing which features were most influential for the Random Forest model.

### Installation & Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/HeartDiseasePrediction.git
   cd HeartDiseasePrediction


