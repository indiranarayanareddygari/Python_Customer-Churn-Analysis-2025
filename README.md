# Python_Customer-Churn-Analysis-2025
 A complete machine learning pipeline project for predicting customer churn using Python.




## ✅ `Customer-Churn-Analysis-2025`

# 🧠 Customer Churn Analysis using Python (2025)

This project builds an end-to-end machine learning pipeline to predict customer churn in a telecom company. The goal is to identify customers likely to cancel their subscription and help the business take proactive retention steps.

---

## 📁 Project Structure

```

├── data/                # Original dataset
├── images/              # EDA visualizations
├── notebooks/           # Jupyter Notebook
├── scripts/             # Reusable Python script
├── requirements.txt     # Required packages
└── README.md            # Project overview (this file)

````

---

## 📊 Dataset Source

- Dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Source: IBM Sample Data
- Size: ~7K customer records
- Target variable: `Churn` (Yes/No)

---

## 🔍 Project Workflow

This project is divided into the following key modules:

---

### 1️⃣ Data Loading

**Why?**  
Loading the dataset is the first step in any data science workflow. It helps us view the structure and content.

**Key Code:**
```python
import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
````

---

### 2️⃣ Data Preprocessing (Cleaning & Type Conversion)

**Why?**
Raw data often contains missing values, wrong formats, or dirty entries that need correction before modeling.

**Tasks Done:**

* Converted `TotalCharges` from string to float
* Removed rows with missing `TotalCharges`
* Converted `SeniorCitizen` from binary (0/1) to categorical (`Yes`/`No`)

**Key Code:**

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
```

---

### 3️⃣ Exploratory Data Analysis (EDA)

**Why?**
EDA helps uncover relationships, patterns, and distributions in the data to inform feature selection and modeling.

**Tasks Done:**

* Countplots for `Churn` vs categorical features
* Histograms + KDE for `tenure`, `MonthlyCharges`, `TotalCharges`
* Boxplots to detect outliers

**Key Code:**

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='Churn')
plt.title("Churn Distribution")
plt.show()

# Example for a numerical feature
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, multiple='stack')
```

📸 Visuals saved under: `images/`

---

### 4️⃣ Feature Engineering & Encoding

**Why?**
Machine learning models need numerical input. This step transforms features into a suitable format.

**Tasks Done:**

* Dropped `customerID`
* Label encoded binary features (`Yes`/`No` → `1`/`0`)
* One-hot encoded multi-category features (e.g., `Contract`, `PaymentMethod`)

**Key Code:**

```python
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', ...])
```

---

### 5️⃣ Train-Test Split

**Why?**
Separating training and testing datasets is essential to evaluate model performance on unseen data.

**Key Code:**

```python
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 6️⃣ Model Training & Evaluation

**Why?**
This step trains the model and checks how well it performs in predicting churn.

**Model Used:** Logistic Regression

**Key Code:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## 📌 Project Highlights

* 🔍 Performed EDA with detailed visualizations
* 🧼 Cleaned and transformed real-world messy data
* 🛠️ Used both Label and One-Hot Encoding
* 🤖 Built a baseline churn prediction model
* 📝 All plots and results saved for future reuse

---

## 📒 Jupyter Notebook

📂 Navigate to: 'Customer_churn_analysis_2025-checkpoint.ipynb'



## ✍️ Author

**Indira \(https://www.linkedin.com/in/indira-narayanareddygari-analyst061294/)**
**Data Analyst |Power BI  & SQL Enthusiast | Python & ML Explorer**


