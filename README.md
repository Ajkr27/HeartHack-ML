


```markdown
# 💓 HeartIQ: Machine Learning Based Heart Disease Prediction

**HeartIQ** is a machine learning-based project designed to predict the presence of heart disease using various health parameters. The project compares the performance of multiple classification models to identify the most accurate one for early detection of heart disease.

### 🧠 **Objective**
The primary goal of this project is to predict the presence or absence of heart disease based on a set of health features. By comparing the performance of several machine learning algorithms, we aim to determine which one gives the best predictive accuracy for real-world use.

---

### 📊 **Project Highlights**
- Preprocessed and cleaned the real-world **Heart Disease UCI Dataset**.
- Implemented and compared **8 different classification models**, including:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - XGBoost
  - Neural Network (MLP)
- Achieved the highest accuracy of **90.16% using Random Forest**.
- Visualized model performance using **bar plots, pie charts, and accuracy comparison graphs**.
- Evaluated models based on accuracy scores to select the most effective one for heart disease prediction.

---

### 🛠 **Technologies Used**
- **Python** (Core Programming Language)
- **Pandas** and **NumPy** (Data Processing)
- **Matplotlib** and **Seaborn** (Data Visualization)
- **Scikit-learn** (Machine Learning Algorithms)
- **XGBoost** (Advanced Gradient Boosting)
- **Neural Networks** (MLP)

---

### 🗂 **Dataset Information**
- **Source**: [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Target**: The target variable is `target`, where:
  - `1` indicates heart disease
  - `0` indicates no heart disease
- **Features** include: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, max heart rate, exercise-induced angina, and more.

---

### 📈 **Model Accuracy Comparison**

| Algorithm              | Accuracy (%) |
|------------------------|--------------|
| Logistic Regression    | 85.25        |
| Naive Bayes            | 85.25        |
| Support Vector Machine | 81.97        |
| K-Nearest Neighbors    | 67.21        |
| Decision Tree          | 81.97        |
| **Random Forest**      | **90.16**    |
| XGBoost                | 81.97        |
| Neural Network (MLP)   | 83.61        |

---

### 🏁 **How to Run**

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/heartiq.git
   cd heartiq
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook heart_disease_prediction.ipynb
   ```

---

### 🔍 **Future Work**
- **Hyperparameter Tuning**: Improve model performance with techniques like GridSearchCV.
- **Model Deployment**: Deploy the best-performing model using platforms like Streamlit or Flask.
- **Real-Time Prediction**: Integrate the prediction model into real-time health monitoring systems.

---

### 📎 **Acknowledgments**
- The **Heart Disease UCI dataset** is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
- Inspired by real-world healthcare applications and predictive analytics in medicine.

---

### 📄 **License**
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

