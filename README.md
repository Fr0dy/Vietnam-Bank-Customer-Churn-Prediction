
# 🏦 Vietnam Bank Churn Prediction Project

## ✨ Introduction

This project focuses on analyzing customer data from a bank in Vietnam to identify factors leading to customer churn and to build a predictive model. The objective is to proactively identify high-risk customers, allowing the bank to implement timely interventions, retain customers, and optimize business strategies.

## 📊 Dataset

The `Vietnam_Bank_Churn.csv` dataset contains detailed information about bank customers, including variables such as age, gender, balance, credit score, income, marital status, number of cards, activity level, and the target variable `exit` (0: not churned, 1: churned).

## 🚀 Analysis & Model Building Steps

The project is executed through the following key steps:

### 1. 🛠️ Setup & Import Libraries

Essential libraries like `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` are installed and imported for data analysis and model building.

### 2. 📁 Data Loading & PII Masking

*   Data is loaded from the `Vietnam_Bank_Churn.csv` file.
*   PII (Personally Identifiable Information) Masking techniques are applied to protect sensitive customer information (`full_name`, `id`) by hashing or partially masking them.
*   A clean DataFrame `df_clean` is created after dropping the original sensitive columns.

### 3. 🔍 Exploratory Data Analysis (EDA)

*   **Churn (Exit) Distribution**: Analyze the overall customer churn rate.
*   **Deep Dive: Zero Balance & Urban Insight**: Evaluate the relationship between zero balance and churn rate. (Note: Data type conversion for the 'balance' column was fixed to prevent Key Errors).
*   **Correlation Heatmap**: Visualize relationships between numerical variables to identify factors strongly influencing churn.

### 4. ⚙️ Feature Engineering & Data Preprocessing

*   **Categorical Variable Handling**: `LabelEncoder` is used to convert textual categorical variables (e.g., `gender`, `occupation`) into numerical form.
*   **Missing Value Imputation**: Remaining numerical columns are filled with 0 if missing values exist.
*   **Data Splitting**: The dataset is split into training and testing sets (80/20 ratio).
*   **Standard Scaling**: `StandardScaler` is applied to numerical features to bring them to a common scale, aiding model performance.

### 5. 🎯 Build Baseline Churn Prediction Model (Random Forest)

*   A **Random Forest Classifier** algorithm is used to build the predictive model. Random Forest is chosen for its ability to handle non-linear data well and provide feature importance insights.
*   **Model Evaluation**: Metrics suchs as `Accuracy`, `ROC-AUC Score`, and `Classification Report` are used to assess the model's performance.
*   **Feature Importance**: Identify the most critical factors influencing customer churn.

### 6. 📝 Export Insights Report

*   A list of high-risk customers (based on `risk_score`) is identified and exported to `High_Risk_Churn_Segment.csv`. This file can be utilized by the customer service team for retention campaigns.

## ✅ Key Results & Insights

*   **Average Churn Rate**: (See results from notebook)
*   **Impact of Zero Balance**: Customers with a zero balance have a lower churn rate (0%) compared to those with a positive balance (18%), which requires further business investigation.
*   **Random Forest Model Performance**:
    *   Accuracy: (See results from notebook)
    *   ROC-AUC Score: (See results from notebook)
    *   Classification Report provides detailed precision, recall, and f1-score for both classes.
*   **Top Feature Importance**: The top features influencing churn are visualized, helping the bank focus on the most impactful factors.

## 🏃 How to Run the Project

1.  **Download Notebook**: Clone this repository or download the `.ipynb` file.
2.  **Upload Data**: Place the `Vietnam_Bank_Churn.csv` file in the same directory as the notebook or upload it to Colab.
3.  **Run Cells**: Execute the cells in the notebook sequentially. Analysis results and the model will be displayed, and the `High_Risk_Churn_Segment.csv` report file will be generated.

## 💻 Technologies Used

*   Python
*   Pandas
*   NumPy
*   Matplotlib
*   Seaborn
*   Scikit-learn
*   `hashlib` (for PII masking)
