# Titanic Data Science Project — BCS 404

**Accra Technical University — Department of Computer Science**
**Course:** BCS 404: Introduction to Data Science with Python
**Lecturer:** Dr. Joseph Dadzie
**Academic Year:** 2025/2026, Second Semester

## Overview

This repository contains the project work for BCS 404, applying exploratory data analysis, statistical analysis, and machine learning to the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data). The goal is to explore the factors associated with passenger survival aboard the RMS Titanic and to build a Logistic Regression model that predicts survival.

## Repository Structure

```
├── Titanic_Data_Science_Project.ipynb   # Main Jupyter Notebook (all tasks, with outputs)
├── analysis.py                          # Equivalent standalone Python script
├── data/
│   ├── titanic.csv                      # Original dataset
│   └── titanic_clean.csv                # Cleaned dataset (output of Task 2)
├── figures/                             # Saved PNG charts (Task 3)
├── report/
│   ├── Titanic_Project_Report.docx      # Full project report (Word)
│   └── Titanic_Project_Report.pdf       # Full project report (PDF)
├── requirements.txt                     # Python dependencies
└── README.md
```

## How to Run

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Titanic_Data_Science_Project.ipynb
   ```
   Or run the standalone script:
   ```bash
   python analysis.py
   ```

## Project Tasks Covered

1. **Data Acquisition** — loading the dataset, inspecting shape, columns, dtypes, and first observations.
2. **Data Cleaning** — detecting and handling missing values (Age, Embarked, Cabin) and checking for duplicates.
3. **Data Visualisation** — histogram, bar chart, boxplot, scatter plot, correlation heatmap, and pairplot.
4. **Statistical Analysis** — descriptive statistics, frequency distributions, and correlation analysis.
5. **Machine Learning** — a Logistic Regression classifier predicting survival, evaluated with accuracy, a confusion matrix, and a classification report.
6. **Discussion and Conclusion** — summary of findings, statistical insights, model performance, limitations, and recommendations (see full report).

## Key Results

- Cleaned dataset: 891 rows, 0 missing values (Cabin dropped; Age/Embarked imputed).
- Strongest positive correlation: `SibSp` & `Parch` (r = 0.415).
- Strongest negative correlation: `Pclass` & `Fare` (r = -0.549).
- Logistic Regression test accuracy: **~80.4%**.

## Dataset Source

Titanic dataset, Kaggle: https://www.kaggle.com/competitions/titanic/data

## Author

[Your Full Name] — [Your Index Number]
