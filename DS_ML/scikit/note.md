# Scikit-learn Notes: One-Hot Encoding and Train/Test Split

## 1. Converting Categorical Data into Numbers

Many machine learning algorithms **cannot work with text (categorical data)**. Therefore, we must convert categorical values into numerical values before training the model.

### Example Dataset

```python
import pandas as pd

X = pd.DataFrame({
    "Make": ["Toyota", "Honda", "BMW"],
    "Colour": ["Red", "Blue", "Red"],
    "Doors": [4, 2, 4],
    "Odometer": [150000, 80000, 120000]
})
```

| Make   | Colour | Doors | Odometer |
| ------ | ------ | ----: | -------: |
| Toyota | Red    |     4 |   150000 |
| Honda  | Blue   |     2 |    80000 |
| BMW    | Red    |     4 |   120000 |

---

## OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder()
```

### Definition

`OneHotEncoder` converts **categorical values** into **binary (0 and 1) columns**.

### Example

Instead of

```
Toyota
Honda
BMW
```

it becomes

| BMW | Honda | Toyota |
| --: | ----: | -----: |
|   0 |     0 |      1 |
|   0 |     1 |      0 |
|   1 |     0 |      0 |

Each category gets its own column.

---

## Selecting the Categorical Columns

```python
categorical_features = ["Make", "Colour", "Doors"]
```

This tells Scikit-learn which columns should be encoded.

Only these columns will be transformed.

---

## ColumnTransformer

```python
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    [
        (
            "one_hot",
            one_hot,
            categorical_features
        )
    ],
    remainder="passthrough"
)
```

### Purpose

`ColumnTransformer` applies different transformations to different columns of a dataset.

### General Syntax

```python
ColumnTransformer(
    [
        (
            name,
            transformer,
            columns
        )
    ],
    remainder="passthrough"
)
```

### Meaning of Each Part

```python
(
    "one_hot",
    one_hot,
    categorical_features
)
```

| Part                   | Meaning                                                                          |
| ---------------------- | -------------------------------------------------------------------------------- |
| `"one_hot"`            | A name (label) for this transformation. It can be almost any descriptive string. |
| `one_hot`              | The `OneHotEncoder` object that performs the encoding.                           |
| `categorical_features` | The list of columns to encode.                                                   |

---

## remainder="passthrough"

```python
remainder="passthrough"
```

### Purpose

Keeps all columns that are **not transformed**.

Example:

```
Columns:

Make
Colour
Doors
Odometer
```

Only

```
Make
Colour
Doors
```

are encoded.

```
Odometer
```

remains unchanged.

Without `remainder="passthrough"`, the `Odometer` column would be dropped.

---

## fit_transform()

```python
transformed_X = transformer.fit_transform(X)
```

This performs two operations:

### fit()

Learns all categories from the data.

Example:

```
Make:
Toyota
Honda
BMW

Colour:
Red
Blue

Doors:
2
4
```

### transform()

Converts those categories into one-hot encoded columns.

---

## Summary

* `OneHotEncoder()` creates an encoder for categorical data.
* `ColumnTransformer()` chooses which columns should use which transformer.
* `"one_hot"` is just a label.
* `remainder="passthrough"` keeps the remaining columns unchanged.
* `fit_transform()` learns the categories (`fit`) and converts them (`transform`).

---

# 2. Train/Test Split and Random Seed

Machine learning models should **not be trained on the entire dataset**.

Instead, the dataset is divided into:

* Training data
* Testing data

The training data teaches the model.

The testing data checks how well the model performs on unseen data.

---

## train_test_split()

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    transformed_X,
    y,
    test_size=0.2
)
```

### Parameters

* `transformed_X` → Input features.
* `y` → Target values.
* `test_size=0.2` → 20% testing data and 80% training data.

---

## Why is the Data Shuffled?

Suppose the data is ordered.

```
Cheap
Cheap
Cheap
Medium
Medium
Expensive
Luxury
```

If we always take the last 20% for testing, the model may never learn about luxury cars.

Therefore, Scikit-learn randomly shuffles the dataset before splitting it.

---

# Random Seed

```python
import numpy as np

np.random.seed(42)
```

### Definition

A **seed** is the starting value used by the random number generator.

It makes random operations produce the **same results every time the program is run**.

Without a seed:

```
Run 1 → Different split

Run 2 → Different split

Run 3 → Different split
```

With

```python
np.random.seed(42)
```

the train/test split will be exactly the same every time.

---

## Why Use a Seed?

Benefits:

* Makes experiments reproducible.
* Makes debugging easier.
* Allows others to get the same results.
* Ensures consistent train/test splits.

---

## Why is 42 Used?

There is nothing special about 42.

Any integer works.

Examples:

```python
np.random.seed(1)
np.random.seed(10)
np.random.seed(100)
```

The number 42 is simply a popular convention among programmers.

---

## model.fit()

```python
model.fit(X_train, y_train)
```

### Purpose

Trains the machine learning model.

The model learns the relationship between:

* `X_train` → Input features.
* `y_train` → Correct answers (target values).

After training, the model can make predictions on new, unseen data.

---

## Modern Scikit-learn Style

Instead of

```python
np.random.seed(42)

train_test_split(...)
```

it is more common to write

```python
X_train, X_test, y_train, y_test = train_test_split(
    transformed_X,
    y,
    test_size=0.2,
    random_state=42
)
```

Using `random_state=42` makes the split reproducible without setting NumPy's global random seed.

---

# Quick Summary

### One-Hot Encoding

* Converts categorical values into numerical binary columns.
* `OneHotEncoder()` performs the encoding.
* `ColumnTransformer()` applies transformations to selected columns.
* `remainder="passthrough"` keeps the remaining columns.
* `fit_transform()` learns categories and transforms the data.

### Train/Test Split

* Splits data into training and testing sets.
* `test_size=0.2` means 80% training and 20% testing.
* Data is randomly shuffled before splitting.

### Random Seed

* Controls randomness.
* Produces the same random results every run.
* Makes experiments reproducible.
* `random_state=42` is the preferred Scikit-learn approach.

### Model Training

```python
model.fit(X_train, y_train)
```

Trains the model using the training data so it can learn patterns and make predictions.

# score() in Scikit-learn

After training a machine learning model, we need to evaluate how well it performs.

Scikit-learn provides the `score()` method for this purpose.

## Syntax

```python
model.score(X_test, y_test)
```

### Parameters

* `X_test` → The testing features (input data).
* `y_test` → The correct target values (actual answers).

The model uses `X_test` to make predictions and compares them with `y_test`.

---

## Example

```python
# Train the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)

print(score)
```

**Possible Output**

```text
0.92
```

---

## What Does the Score Mean?

The meaning of `score()` depends on the type of machine learning model.

### 1. Classification Models

For classification models (e.g., `RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier`), `score()` returns **accuracy**.

**Formula**

```text
Accuracy = (Number of Correct Predictions) / (Total Predictions)
```

Example:

Suppose the model predicts 10 samples.

```text
Correct Predictions = 9
Total Predictions = 10
```

Then

```text
Accuracy = 9 / 10 = 0.9 = 90%
```

So

```python
model.score(X_test, y_test)
```

returns

```text
0.9
```

which means **90% accuracy**.

---

### 2. Regression Models

For regression models (e.g., `RandomForestRegressor`, `LinearRegression`), `score()` returns the **R² Score (Coefficient of Determination)**.

The R² score measures how well the model explains the variation in the target values.

|    R² Score | Meaning                                                 |
| ----------: | ------------------------------------------------------- |
|         1.0 | Perfect predictions                                     |
|         0.0 | Model performs no better than predicting the average    |
| Less than 0 | Model performs worse than simply predicting the average |

Example:

```python
model.score(X_test, y_test)
```

Output

```text
0.87
```

This means the model explains about **87% of the variation** in the target values.

---

## Workflow

```text
Prepare Data
      │
      ▼
Split Data
      │
      ▼
Train Model
model.fit(X_train, y_train)
      │
      ▼
Evaluate Model
model.score(X_test, y_test)
      │
      ▼
Model Performance
```

---

## Difference Between fit() and score()

| Method                        | Purpose                                                               |
| ----------------------------- | --------------------------------------------------------------------- |
| `model.fit(X_train, y_train)` | Trains (learns from) the training data.                               |
| `model.score(X_test, y_test)` | Evaluates how well the trained model performs on unseen testing data. |

---

## Important Notes

* Always call `fit()` before `score()`.
* Use the **testing data** (`X_test`, `y_test`) to evaluate the model.
* A higher score generally indicates better performance, but the interpretation depends on the model type.
* `score()` is a quick evaluation method. For deeper analysis, Scikit-learn also provides metrics such as:

  * `accuracy_score()`
  * `precision_score()`
  * `recall_score()`
  * `f1_score()`
  * `mean_squared_error()`
  * `r2_score()`

---

## Quick Summary

```python
# Train the model
model.fit(X_train, y_train)

# Evaluate the model
model.score(X_test, y_test)
```

* **`fit()`** → Learns patterns from the training data.
* **`score()`** → Measures how well the trained model performs on unseen test data.
* **Classification:** `score()` returns **Accuracy**.
* **Regression:** `score()` returns the **R² Score**.

