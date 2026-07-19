import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

FIG_DIR = "figures"

# ---------------- Task 1: Data Acquisition ----------------
df = pd.read_csv("data/titanic.csv")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())
print(df.dtypes)

# ---------------- Task 2: Data Cleaning ----------------
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

df_clean = df.copy()
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
df_clean.drop(columns=['Cabin'], inplace=True)
df_clean.drop_duplicates(inplace=True)
print("\nAfter cleaning missing values:\n", df_clean.isnull().sum())
print("Cleaned shape:", df_clean.shape)

# ---------------- Task 3: Visualisation ----------------
plt.figure()
plt.hist(df_clean['Age'], bins=30, color='steelblue', edgecolor='black')
plt.title("Distribution of Passenger Ages")
plt.xlabel("Age (years)")
plt.ylabel("Frequency")
plt.savefig(f"{FIG_DIR}/1_age_histogram.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
sns.countplot(x='Pclass', data=df_clean, palette='viridis', hue='Pclass', legend=False)
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.savefig(f"{FIG_DIR}/2_pclass_bar.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
sns.boxplot(x='Pclass', y='Age', data=df_clean, hue='Pclass', palette='Set2', legend=False)
plt.title("Age Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age (years)")
plt.savefig(f"{FIG_DIR}/3_age_by_pclass_box.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
plt.scatter(df_clean['Age'], df_clean['Fare'], alpha=0.5, c='darkorange', edgecolor='k')
plt.title("Age vs Fare")
plt.xlabel("Age (years)")
plt.ylabel("Fare (£)")
plt.savefig(f"{FIG_DIR}/4_age_vs_fare_scatter.png", dpi=150, bbox_inches='tight')
plt.close()

num_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
plt.figure(figsize=(8, 6))
corr = df_clean[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title("Correlation Heatmap of Numerical Variables")
plt.savefig(f"{FIG_DIR}/5_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

pp = sns.pairplot(df_clean[num_cols], hue='Survived', palette='husl', diag_kind='hist')
pp.fig.suptitle("Pairplot of Selected Numerical Variables", y=1.02)
pp.savefig(f"{FIG_DIR}/6_pairplot.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nCorrelation matrix:\n", corr)

# ---------------- Task 4: Statistical Analysis ----------------
print("\nDescriptive statistics:\n", df_clean[num_cols].describe())
print("\nSurvived counts:\n", df_clean['Survived'].value_counts())
print("\nSex counts:\n", df_clean['Sex'].value_counts())
print("\nEmbarked counts:\n", df_clean['Embarked'].value_counts())

corr_unstack = corr.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_unstack[corr_unstack < 1].drop_duplicates()
print("\nTop correlations by magnitude:\n", corr_pairs.head(10))

signed = corr.unstack()
signed = signed[signed != 1].drop_duplicates()
print("\nStrongest positive:", signed.idxmax(), signed.max())
print("Strongest negative:", signed.idxmin(), signed.min())

# ---------------- Task 5: Machine Learning ----------------
ml_df = df_clean.copy()
le_sex = LabelEncoder()
ml_df['Sex_enc'] = le_sex.fit_transform(ml_df['Sex'])
le_emb = LabelEncoder()
ml_df['Embarked_enc'] = le_emb.fit_transform(ml_df['Embarked'])

features = ['Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_enc']
X = ml_df[features]
y = ml_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Did Not Survive', 'Survived'], yticklabels=['Did Not Survive', 'Survived'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{FIG_DIR}/7_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

df_clean.to_csv("data/titanic_clean.csv", index=False)
print("\nDone.")
