
## Manual Linear Regression (Normal Equation)
Prepare data (only first 7 rows from Asian cars, 2 features):
```python
X = df.loc[df["origin"] == "Asia", ["vehicle_weight", "model_year"]].head(7).to_numpy()
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
```
## 2. Compute normal equation:

- w=(XTX)−1XTy
```pyhton
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
w = XTX_inv @ X.T @ y
```
## 3. Sum of weights:
w_sum = w.sum()
print(w_sum)
- w contains the regression coefficients for vehicle_weight and model_year.

- w_sum is just the sum of these two coefficients — a quick check or summary statistic.


##  Handle Missing Values
```python
print("Missing values per column:")
print(df.isnull().sum())

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':  # categorical
        df[col] = df[col].fillna('NA')
    else:  # numerical
        df[col] = df[col].fillna(0.0)

print("\nMissing values after filling:")
print(df.isnull().sum())
```
## Basic Statistics and Correlations
```python
# Mode / Median example
df['industry'].mode()[0]
df['horsepower'].median()

# Correlation matrix (numerical features)
corr_matrix = df.corr(numeric_only=True)

# Find strongest correlation excluding self
corr_unstacked = corr_matrix.unstack().sort_values(ascending=False)
corr_unstacked = corr_unstacked[corr_unstacked < 1]
biggest_corr = corr_unstacked.idxmax()
corr_value = corr_unstacked.max()
print(f"Biggest correlation: {biggest_corr} = {corr_value:.2f}")
```
## Mutual Information for Categorical Features
```python
from sklearn.feature_selection import mutual_info_classif

categorical_cols = ['industry', 'location', 'lead_source', 'employment_status']

# Convert to numeric codes
data = df[categorical_cols + ['converted']].dropna()
for col in categorical_cols:
    data[col] = data[col].astype('category').cat.codes

X = data[categorical_cols]
y = data['converted']
mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=1)

mi_df = pd.DataFrame({'Feature': categorical_cols, 'MI Score': mi_scores}).sort_values(by='MI Score', ascending=False)
print(mi_df)
```
## Random Forest Model (Numeric Features)
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
X = df[features].fillna(df[features].median())
y = df['converted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")
```
## Logistic Regression with Hyperparameter Tuning (GridSearchCV)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X = df[features].fillna(0)
y = df['converted']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

param_grid = {'logreg__C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print("Best C:", grid.best_params_['logreg__C'])
print("Best CV accuracy:", grid.best_score_)

```

## Logistic Regression Single Feature Testing
```python
X = df[['number_of_courses_viewed']].fillna(0)
y = df['converted']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for C in [0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"C={C}: Accuracy={acc:.2f}")
```

# Machine Learning Model Training & Evaluation Reference
## 1️⃣ Split the Dataset (Train / Validation / Test)
from sklearn.model_selection import train_test_split

## Split: 60% train, 20% validation, 20% test
```python
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=1)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=1)

print(f"Train set: {len(train_df)}")
print(f"Validation set: {len(val_df)}")
print(f"Test set: {len(test_df)}")
```


2️⃣ ROC AUC Feature Importance for Numerical Variables

```python
from sklearn.metrics import roc_auc_score

num_vars = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
auc_scores = {}

for col in num_vars:
    auc = roc_auc_score(train_df['converted'], train_df[col])
    if auc < 0.5:
        auc = roc_auc_score(train_df['converted'], -train_df[col])
    auc_scores[col] = auc

for col, auc in auc_scores.items():
    print(f"{col}: {auc:.3f}")

best_feature = max(auc_scores, key=auc_scores.get)
print(f"\nBest feature: {best_feature} (AUC = {auc_scores[best_feature]:.3f})")
```

## 3️⃣ Train Logistic Regression Model (with One-Hot Encoding)
```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

y_train = train_df['converted']
y_val = val_df['converted']

feature_cols = ['lead_source', 'industry', 'number_of_courses_viewed', 'annual_income',
                'employment_status', 'location', 'interaction_count', 'lead_score']

train_dicts = train_df[feature_cols].to_dict(orient='records')
val_dicts = val_df[feature_cols].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred_val = model.predict_proba(X_val)[:, 1]

auc_val = roc_auc_score(y_val, y_pred_val)
print(f"Validation AUC: {auc_val:.3f}")
```

## 4️⃣ Precision and Recall vs Threshold
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

thresholds = np.linspace(0, 1, 101)
precisions, recalls = [], []

for t in thresholds:
    preds = (y_pred_val >= t).astype(int)
    precisions.append(precision_score(y_val, preds))
    recalls.append(recall_score(y_val, preds))

plt.figure(figsize=(8,5))
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.show()

# Intersection threshold
diff = np.abs(np.array(precisions) - np.array(recalls))
intersection_threshold = thresholds[np.argmin(diff)]
print(f"Precision and Recall intersect around threshold = {intersection_threshold:.2f}")
```
5️⃣ F1 Score vs Threshold
```python
f1_scores = 2 * np.array(precisions) * np.array(recalls) / (np.array(precisions) + np.array(recalls) + 1e-10)

plt.figure(figsize=(8,5))
plt.plot(thresholds, f1_scores, color='purple', label='F1 Score')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.legend()
plt.grid(True)
plt.show()

best_idx = np.argmax(f1_scores)
print(f"Best F1 = {f1_scores[best_idx]:.3f} at threshold = {thresholds[best_idx]:.2f}")
```
## 6️⃣ 5-Fold Cross-Validation (Single C value)
```Python
from sklearn.model_selection import KFold

df_full_train = train_df.copy()
feature_cols = ['lead_source', 'industry', 'number_of_courses_viewed', 'annual_income',
                'employment_status', 'location', 'interaction_count', 'lead_score']

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
auc_scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train_fold = df_full_train.iloc[train_idx]
    df_val_fold = df_full_train.iloc[val_idx]
    
    y_train_fold = df_train_fold['converted']
    y_val_fold = df_val_fold['converted']
    
    dv = DictVectorizer(sparse=False)
    X_train_fold = dv.fit_transform(df_train_fold[feature_cols].to_dict(orient='records'))
    X_val_fold = dv.transform(df_val_fold[feature_cols].to_dict(orient='records'))
    
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train_fold, y_train_fold)
    
    y_pred = model.predict_proba(X_val_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, y_pred)
    auc_scores.append(auc)

print(f"AUCs: {[round(a, 3) for a in auc_scores]}")
print(f"Mean AUC: {np.mean(auc_scores):.3f}, Std: {np.std(auc_scores):.3f}")
```
## 7️⃣ Cross-Validation for Best C Parameter
```pyhton

C_values = [0.000001, 0.001, 1]
results = {}

for C in C_values:
    auc_scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train_fold = df_full_train.iloc[train_idx]
        df_val_fold = df_full_train.iloc[val_idx]
        
        y_train_fold = df_train_fold['converted']
        y_val_fold = df_val_fold['converted']
        
        dv = DictVectorizer(sparse=False)
        X_train_fold = dv.fit_transform(df_train_fold[feature_cols].to_dict(orient='records'))
        X_val_fold = dv.transform(df_val_fold[feature_cols].to_dict(orient='records'))
        
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_fold, y_train_fold)
        
        y_pred = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred)
        auc_scores.append(auc)
    
    results[C] = (round(np.mean(auc_scores), 3), round(np.std(auc_scores), 3))

for C, (mean_auc, std_auc) in results.items():
    print(f"C={C}: mean AUC={mean_auc}, std={std_auc}")

best_C = max(results, key=lambda k: results[k][0])
print(f"\nBest C: {best_C} (mean AUC={results[best_C][0]}, std={results[best_C][1]})")
```




