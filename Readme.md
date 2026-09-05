# Student Placement Eligibility Prediction using Logistic Regression

## 📌 Project Overview

This project focuses on predicting whether a student is likely to meet a **placement eligibility criterion** using a **Machine Learning binary classification model**.

A **Logistic Regression** model is developed using student-related academic and skill-based features. The project demonstrates an end-to-end Machine Learning workflow, including data inspection, data quality checking, preprocessing, model training, prediction, evaluation, and feature-effect analysis.

> **Note:** This project is an educational Machine Learning model and should not be considered a real-world placement decision system.

---

## 🎯 Objective

The primary objective of this project is to build a binary classification model using **Logistic Regression** to predict student placement eligibility.

The model predicts:

* `target = 1` → Positive / Event Class
* `target = 0` → Negative / Non-Event Class

---

## 📊 Features Used

The model uses the following six predictors:

| Feature              | Description                   |
| -------------------- | ----------------------------- |
| `cgpa`               | Student's CGPA                |
| `attendance_pct`     | Attendance percentage         |
| `coding_score`       | Coding assessment score       |
| `projects_completed` | Number of projects completed  |
| `internship_months`  | Internship duration in months |
| `backlogs`           | Number of academic backlogs   |

### Target Variable

`target` is the binary target variable representing the placement eligibility class.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Scikit-learn** – Machine Learning
* **Google Colab** – Development environment

---

## 🤖 Machine Learning Algorithm

### Logistic Regression

Logistic Regression is used because this is a **binary classification problem**.

The model estimates the probability that a student belongs to the positive class (`target = 1`).

The implementation uses:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

---

## 🔄 Project Workflow

The project follows these steps:

```text
Dataset
   ↓
Load Dataset
   ↓
Dataset Inspection
   ↓
Missing Value & Duplicate Check
   ↓
Target Class Balance Check
   ↓
Feature Range / Validity Check
   ↓
Define Features & Target
   ↓
80/20 Stratified Train-Test Split
   ↓
StandardScaler
   ↓
Logistic Regression
   ↓
Model Training
   ↓
Predictions & Probabilities
   ↓
Model Evaluation
   ↓
Feature Coefficient Analysis
   ↓
New Student Prediction
```

---

## 🔍 Data Inspection

The notebook performs several checks before training the model:

* Column names
* Data types
* Statistical summary
* Dataset information
* Missing values
* Duplicate rows
* Target class distribution
* Feature ranges
* Minimum and maximum values
* Mean values
* Number of unique values

This helps understand the dataset and identify potential data-quality issues.

---

## 🧹 Data Preprocessing

### Standardization

The six numerical predictors have different scales. Therefore, **StandardScaler** is used before Logistic Regression.

The preprocessing and model are combined using a Scikit-learn `Pipeline`:

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])
```

Using a pipeline ensures that the scaler is fitted only using the training data, helping prevent **data leakage** from the test set.

---

## ✂️ Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

A stratified split is used to preserve the target class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

## 📈 Model Evaluation

The trained Logistic Regression model is evaluated using multiple classification metrics:

### Accuracy

Measures the overall proportion of correctly classified observations.

### Precision

Measures how many of the students predicted as positive actually belong to the positive class.

### Recall

Measures how many of the actual positive-class students were correctly identified.

### F1-Score

Provides a balance between Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

### Confusion Matrix

The confusion matrix provides:

* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)
* True Positives (TP)

---

## 📊 Visualizations

The project generates the following visualizations:

### 1. Target Class Distribution

Shows the distribution of the two target classes.

### 2. Confusion Matrix

Visualizes the classification results of the Logistic Regression model.

### 3. ROC Curve

Displays the relationship between the True Positive Rate and False Positive Rate and reports the ROC-AUC score.

### 4. Feature Coefficient Visualization

Displays the Logistic Regression coefficients for the six predictors.

---

## 🔬 Feature Effect Analysis

The project analyzes the Logistic Regression coefficients to understand how each standardized feature is associated with the positive class.

Because the predictors are standardized:

* **Positive coefficient** → increases the odds of `target = 1`
* **Negative coefficient** → decreases the odds of `target = 1`

The project also calculates **odds ratios** using:

```python
np.exp(coefficient)
```

This provides an interpretable way to examine the estimated effect of each feature.

---

## 👨‍🎓 Example Student Prediction

The notebook demonstrates prediction for a new student with the following example values:

| Feature            |    Value |
| ------------------ | -------: |
| CGPA               |      8.2 |
| Attendance         |      85% |
| Coding Score       |        8 |
| Projects Completed |        4 |
| Internship         | 6 months |
| Backlogs           |        0 |

The trained model generates:

* Predicted class
* Probability of `target = 1`
* Eligibility interpretation based on the predicted class

---

## 📁 Project Structure

```text
Student-Placement-Eligibility-Prediction/
│
├── Student_Placement_Eligibility_Logistic_Regression.ipynb
├── dataset_07_student_placement_eligibility.csv
└── README.md
```

> If the dataset cannot be shared publicly, keep the CSV out of the repository and provide appropriate instructions for obtaining or using the dataset.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Student-Placement-Eligibility-Prediction
```

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## ▶️ How to Run

### Option 1: Google Colab

1. Open the `.ipynb` notebook in Google Colab.
2. Upload the required CSV dataset.
3. Run the cells sequentially.
4. Review the generated metrics, plots, and predictions.

### Option 2: Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Student_Placement_Eligibility_Logistic_Regression.ipynb
```

Run the notebook cells sequentially.

---

## 📋 Model Configuration

| Parameter          | Value                 |
| ------------------ | --------------------- |
| Algorithm          | Logistic Regression   |
| Preprocessing      | StandardScaler        |
| Train/Test Split   | 80/20                 |
| Split Type         | Stratified            |
| Random State       | 42                    |
| Maximum Iterations | 1000                  |
| Target Type        | Binary Classification |

---

## ⚠️ Limitations

This project has several limitations:

1. The dataset contains only six predictors.
2. Real-world placement decisions can depend on additional factors such as communication skills, aptitude scores, interview performance, and company-specific requirements.
3. A single train/test split can produce performance estimates that vary depending on the split.
4. Logistic Regression assumes a linear relationship between predictors and the log-odds of the target.
5. The dataset may not represent students from different colleges, academic years, or recruitment environments.

Therefore, the model should be considered an **educational baseline rather than a real-world placement decision system**.

---

## 🚀 Possible Improvements

Future improvements could include:

* Using cross-validation for more robust performance estimation.
* Collecting a larger and more diverse dataset.
* Adding additional academic and placement-related features.
* Optimizing the classification threshold.
* Comparing Logistic Regression with other technically appropriate classification algorithms.
* Performing more extensive feature engineering.
* Evaluating model performance on an independent dataset.

---

## 📚 Key Learning Outcomes

Through this project, I gained practical experience in:

* Binary classification
* Logistic Regression
* Data preprocessing
* Feature scaling
* Train-test splitting
* Stratified sampling
* Data quality analysis
* Classification metrics
* Confusion Matrix
* ROC Curve and ROC-AUC
* Feature coefficient interpretation
* Odds-ratio analysis
* Probability-based prediction
* Scikit-learn Pipelines
* End-to-end Machine Learning workflow

---

## 👨‍💻 Project Author

**Kishore G**

Machine Learning | Python | Data Science | Artificial Intelligence

---

## 🙏 Acknowledgement

This project was completed as part of my learning journey with **LearnDepth**.

The project provided an opportunity to apply Machine Learning concepts to a practical student placement eligibility prediction problem.

---

## ⭐ Conclusion

The project demonstrates how **Logistic Regression** can be used as a simple and interpretable baseline for binary student placement eligibility prediction.

The complete workflow covers data inspection, preprocessing, model training, prediction, evaluation, visualization, and interpretation, providing practical experience with an end-to-end Machine Learning project.
