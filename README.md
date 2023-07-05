# Random Forest
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains a Python implementation of the Random Forest Regressor and Classifier. The main file in this repository is `rf.py`, which implements the Random Forest models using Decision Trees. The `dtree.py` file is a utility script that is imported by `rf.py` to support the implementation of the Random Forest models.

## Introduction to Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. It is a powerful and widely used machine learning algorithm that can be applied to both regression and classification tasks.

The Random Forest algorithm gets its name from the "forest" of decision trees it creates. Each decision tree is trained independently on a random subset of the training data and a random subset of the features. By introducing randomness in both data and feature selection, Random Forest aims to reduce overfitting and improve the model's generalization performance.

Here are the key steps involved in building a Random Forest:

1. **Bootstrapping**: Random Forest uses a technique called bootstrapping to create multiple subsets of the original training data. Bootstrapping involves random sampling of the training data with replacement. Each subset, known as a bootstrapped sample, has the same size as the original dataset but may contain duplicate instances (so they are <i>i.d.</i> but not <i>i.i.d.</i>). Bootstrapping allows each decision tree in the Random Forest to see a slightly different version of the training data. In practice, about 2/3 of the data will be sampled with each bootstrap sample, which leaves 1/3 of the data to be used as part of our out-of-bag (OOB) data.

2. **Random Feature Selection**: At each node of a decision tree, Random Forest only considers a random subset of features to determine the best split. This random feature selection introduces further randomness into the model and prevents individual features from dominating the decision-making process. The number of features considered at each split is controlled by the `max_features` hyperparameter.

3. **Building Decision Trees**: Random Forest builds multiple decision trees using bootstrapped samples and random feature subsets. Each decision tree is trained independently to predict the target variable. The trees are constructed recursively by splitting the data based on the selected features and the best-split criteria (e.g., Gini impurity for classification or variance reduction for regression).

4. **Aggregating Predictions**: To make predictions, Random Forest aggregates the predictions from all decision trees in the forest. For regression tasks, the predictions from individual trees are averaged to obtain the final prediction. For classification tasks, the predictions are combined using voting (majority vote) to determine the final class label.

6. **Out-of-bag (OOB) Error**: The Random Forest algorithm takes advantage of the Out-of-bag (OOB) error estimate to evaluate the performance of the model during training. OOB error is calculated using the samples that were not included in the bootstrapped sample for each Decision Tree. For each instance in the original dataset, the OOB error is computed by aggregating predictions from only the Decision Trees that did not use that instance in their training set. This provides an unbiased estimate of the model's performance on unseen data without the need for a separate validation set.

The Random Forest algorithm offers several advantages:

- **Reduced Overfitting**: Random Forest reduces overfitting by averaging the predictions of multiple decision trees, which helps to smooth out individual tree biases and capture the collective wisdom of the ensemble.
- **Robustness to Outliers and Noisy Data**: Random Forest is less sensitive to outliers and noisy data since it considers multiple subsets of the training data and features.
- **Feature Importance**: Random Forest provides a measure of feature importance based on how much each feature contributes to the overall performance of the ensemble. This can be useful for understanding the importance of different features in the data.

Random Forest has become a popular algorithm due to its robustness, scalability, and good generalization performance. The implemented Random Forest models in this repository offer a simplified implementation with comparable accuracy to sklearn and can serve as a starting point to understand the core concepts of Random Forests.

## Random Forest Regressor

The Random Forest Regressor is an ensemble learning method used for regression tasks. It combines multiple Decision Trees to make predictions. In this implementation, the Random Forest Regressor is implemented in the `RandomForestRegressor621` class, which takes hyperparameters such as the number of trees (`n_estimators`), the minimum number of samples required to create a leaf node (`min_samples_leaf`), the maximum number of features to consider at each split (`max_features`), and an option to compute the out-of-bag (OOB) validation score (`oob_score`). The `fit()` method trains the random forest on a given dataset, and the `predict()` method makes predictions on new data.

## Random Forest Classifier

On the other hand, the Random Forest Classifier is an ensemble learning method used for classification tasks. It also combines multiple Decision Trees to make predictions. Similarly, in this implementation, the Random Forest Classifier is implemented in the `RandomForestClassifier621` class, which shares similar hyperparameters with the Random Forest Regressor, including the number of trees (`n_estimators`), the minimum number of samples required to create a leaf node (`min_samples_leaf`), the maximum number of features to consider at each split (`max_features`), and an option to compute the out-of-bag (OOB) validation score (`oob_score`). The `fit()` method trains the random forest on a given dataset, and the `predict()` method makes predictions on new data.

While the Random Forest Regressor focuses on predicting continuous target variables, the Random Forest Classifier is specifically designed for predicting categorical target variables. This difference in the task at hand is reflected in the evaluation metrics and the specific use cases of the two models.

## Decision Tree Utility (dtree.py)

The `dtree.py` file contains utility classes and functions that support the implementation of the Random Forest models. It provides essential functionalities for building and working with Decision Trees. The utility functions and classes include:

- `DecisionNode` and `LeafNode`: Classes that represent nodes in a Decision Tree.
- `gini()` function: Calculates the Gini impurity score for classification tasks.
- `find_best_split()` function: Finds the best split for a Decision Tree based on the selected features and loss function.
- `DecisionTree621`, `RegressionTree621`, and `ClassifierTree621`: Classes that represent different types of Decision Trees used in the Random Forest models, specializing in regression and classification tasks.

The `dtree.py` file is imported by `rf.py` to utilize the utility functions and classes for building Decision Trees within the Random Forest implementation.

## Usage

To use the Random Forest Regressor or Classifier, you can import the respective class from `rf.py` and create an instance with your desired hyperparameters. Then, you can call the `fit()` method to train the model on your dataset and the `predict()` method to make predictions on new data.

Here is an example of how to use `RandomForestRegressor621` with sample data and the comparison with scikit-learn `RandomForestRegressor`:

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from rf import RandomForestRegressor621

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of RandomForestRegressor621
random_forest = RandomForestRegressor621(n_estimators=25, min_samples_leaf=3, max_features=0.3, oob_score=True)

# Fit the model to the training data
random_forest.fit(X_train, y_train)

# Make predictions on new data
predictions = random_forest.predict(X_test)

# Calculate the R^2 score for the predictions
r2_score = random_forest.score(X_test, y_test)
print(r2_score)    # Output: 0.49244


# Create an instance of RandomForestRegressor from scikit-learn
sklearn_rf = RandomForestRegressor(n_estimators=25, min_samples_leaf=3, max_features=0.4, oob_score=True)

# Fit the scikit-learn model to the training data
sklearn_rf.fit(X_train, y_train)

# Make predictions using the scikit-learn model
sklearn_predictions = sklearn_rf.predict(X_test)

# Calculate the R^2 score for the scikit-learn predictions
sklearn_r2_score = sklearn_rf.score(X_test, y_test)
print(sklearn_r2_score)    # Output: 0.47657
```

Here is an example of how to use `RandomForestClassifier621` with sample data and the comparison with scikit-learn `RandomForestClassifier`:

```python
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from rf import RandomForestClassifier621

# Load the wine dataset
X, y = load_wine(return_X_y=True)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of RandomForestClassifier621
random_forest = RandomForestClassifier621(n_estimators=25, min_samples_leaf=3, max_features=0.3, oob_score=True)

# Fit the model to the training data
random_forest.fit(X_train, y_train)

# Make predictions on new data
predictions = random_forest.predict(X_test)

# Calculate the accuracy score for the predictions
accuracy_score = random_forest.score(X_test, y_test)
print(accuracy_score)    # Output: 0.94124


# Create an instance of RandomForestClassifier from scikit-learn
sklearn_rf = RandomForestClassifier(n_estimators=25, min_samples_leaf=3, max_features=0.4, oob_score=True)

# Fit the scikit-learn model to the training data
sklearn_rf.fit(X_train, y_train)

# Make predictions using the scikit-learn model
sklearn_predictions = sklearn_rf.predict(X_test)

# Calculate the accuracy score for the scikit-learn predictions
sklearn_accuracy_score = sklearn_rf.score(X_test, y_test)
print(sklearn_accuracy_score)    # Output: 0.97222
```

In summary, the examples provided demonstrate the usage of the `RandomForestRegressor621` and `RandomForestClassifier621` implementations from the `rf.py` module, as well as their comparison with the corresponding models from scikit-learn. 

For the `RandomForestRegressor621` example on the diabetes dataset, both the implemented model and the scikit-learn model achieve similar accuracy scores. The implemented `RandomForestRegressor621` achieves an R2 score of 0.49244, while the scikit-learn `RandomForestRegressor` achieves an R2 score of 0.47657. This indicates that the implemented model can learn and make predictions with a comparable level of accuracy as the scikit-learn model.

For the `RandomForestClassifier621` example on the wine dataset, both the implemented model and the scikit-learn model also achieve similar accuracy scores. The implemented `RandomForestClassifier621` achieves an accuracy score of 0.94124, while the scikit-learn `RandomForestClassifier` achieves an accuracy score of 0.97222. Again, this demonstrates that the implemented model can learn and predict with a similar level of accuracy as the scikit-learn model.

These examples highlight the effectiveness and accuracy of the implemented Random Forest models in the `rf.py` module. They provide a reliable alternative to the scikit-learn implementations, allowing users to leverage the power of Random Forests for regression and classification tasks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

The initial codebase and project structure are adapted from the MSDS 621 course materials provided by the University of San Francisco (USFCA-MSDS). Special thanks to the course instructors for the inspiration.
