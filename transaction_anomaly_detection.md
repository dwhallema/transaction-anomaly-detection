# Detecting anomalies in credit card transaction data (Python, scikit-learn)

Author: [Dennis W. Hallema](https://www.linkedin.com/in/dennishallema) 

Description: Supervised classification procedure for detecting fraudulous credit card transactions in a large dataset. This procedure compares the performance of four classifiers: Logistic Regression, Kernel Support Vector Classifier, Stochastic Gradient Boosting and Random Forest. 

Dependencies: See `environment.yml`. 

Data: PCA transformed credit card transaction data collected in Europe over the course of two days. This anonymized dataset was created by Worldline and the Machine Learning Group of Université Libre de Bruxelles (http://mlg.ulb.ac.be). 

Disclaimer: Use at your own risk. No responsibility is assumed for a user's application of these materials or related materials. 

References: 

* Dal Pozzolo, A., Caelen, O., Le Borgne, Y-A, Waterschoot, S. \& Bontempi, G. (2014). Learned lessons in credit card fraud detection from a practitioner perspective. Expert Systems with Applications, 41(10), 4915-4928. 

* Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C. \& Bontempi, G. (2018). Credit card fraud detection: a realistic modeling and a novel learning strategy. IEEE Transactions on Neural Networks and Learning Systems, 29(8), 3784-3797.

Content: 

* [Data preparation](#one) 
* [Logistic Regression (LR)](#two) 
* [Kernel SVM classification (SVC)](#three) 
* [Gradient Boosting Model (GBM) classification](#four) 
* [Random Forest (RF) classification](#five) 
* [Model selection and cost-effective optimization](#six) 
* [Conclusion](#seven) 


## Data preparation <a id='one'></a>


```python
# Import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
```


```python
# Load data
df = pd.read_csv('data/creditcard.csv', header=None)
df.describe()
```


```python
# Print data types
print(df.dtypes)
print(df.columns)
```


```python
# Define X,y
X = df.iloc[:,:-2]
y = df.iloc[:,-1]
```


```python
# Plot histogram
yhist = plt.hist(y)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('y Distribution')

# Count transactions
pos = sum(y)
pos_rel = sum(y)/y.shape[0]
print("Number of transactions: {}".format(y.shape[0]))
print("Anomalies: {}".format(pos))
print("Anomalies percentage of all transactions: %.4f%%" % (pos_rel*100))
```

*Summary of the credit card transaction dataset:* 

* The variables are unnamed, because we are not working with original credit card data but with variables that have been orthogonally transformed into uncorrelated variables (principal components). 

* There are 29 variables of type float and 1 variable of type integer. The former are the principal component that we can use as features, and the latter is the binary response variable indicating the transaction anomalies. 

* The number of anomalies is very small compared to the total number of transactions collected over the course of two days. In other words, the dataset is highly unbalanced, and this requires special attention when we build a classifier to predict anomalies in the transaction data. 

## Logistic Regression <a id='two'></a>

Predicting anomalies is a binary classification problem. We assume that a transaction represents either an anomaly (1) or not (0), but never both. Logistic Regression is a good starting point for this type of classification because it is fast, and still allows us to explain what variables are influential. (While this is always the case, note that our variables are PCAs meaning that their influence does not give us any information.) We will follow a step-wise approach: 

1. Split the data into a training set and a testing (or hold-out) set; 
2. Scale and center the data; 
3. Fit an initial classifier: 
    * Use default parameters; 
    * Predict (non)anomalous transactions;  
    * Evaluate initial classifier; 
4. Hyperparameter tuning of classifier with k-fold cross-validation: 
    * Identify optimized parameter set for classifier; 
    * Predict (non)anomalous transactions; 
    * Evaluate optimized classifier. 
    


```python
# Import modules
from inspect import signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, mean_squared_error, average_precision_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
```


```python
# Create training and testing sets
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.3, random_state=21)
```


```python
# Scale and center data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```


```python
# Instantiate classifier
clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=21)

# Fit classifier to the training set
clf.fit(X_train, y_train)
```


```python
# Compute training metrics
accuracy = clf.score(X_train, y_train)

#  Predict labels of test set
train_pred = clf.predict(X_train)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_train, train_pred)
conf_mat = confusion_matrix(y_train.round(), train_pred.round())
clas_rep = classification_report(y_train.round(), train_pred.round())

# Print reports
print('{:=^80}'.format('Initial LR training report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute testing metrics
accuracy = clf.score(X_test, y_test)

# Predict labels of test set
y_pred = clf.predict(X_test)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_test, y_pred)
conf_mat = confusion_matrix(y_test.round(), y_pred.round())
clas_rep = classification_report(y_test.round(), y_pred.round())

# Print reports
print('{:=^80}'.format('Initial LR testing report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```

<table>
  <tr>
    <td></td>
    <td>Prediction: 0</td>
    <td>Prediction: 1</td>
  </tr>
  <tr>
    <td>Actual: 0 </td>
    <td>True negative</td>
    <td>False positive</td>
  </tr>
  <tr>
    <td>Actual: 1</td>
    <td>False negative</td>
    <td>True positive</td>
  </tr>
</table> 

* Precision = tp / (tp + fp) 
* Recall = tp / (tp + fn) 
* F-beta score = 2 * (precision * recall) / (precision + recall) 

The classification report (above) shows that the accuracy of the model is outstanding (close to 1.00). But this does not mean this a good model. Why? The vast majority - 99.83% of the transactions are not marked as anomalies in the dataset. An alternative model would simply classify all values as 0 (not anomalous), and still have an accuracy of 1.00 (98.83% to be exact). Despite the fact that this classifier has a very high accuracy and a very low mean squared error, we need to evaluate the metrics that reflect the fact that this dataset is highly unbalanced. We want to focus particularly on the class of interest: anomalous transactions. The recall rate for anomalies (value 1), is not indeed very high (0.59). While the precision for anomalies (0.89) is a good result for an uncalibrated model, the confusion matrix shows that we incorrectly classified 61 transactions as anomalous.


```python
# Compute predicted probabilities
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Calculate receiver operating characteristics (ROC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k-')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Initial LR Testing\nROC curve')
plt.show()
```


```python
# Compute AUPRC score
average_precision = average_precision_score(y_test, y_pred_prob)
print("AUPRC: {}".format(average_precision))

# Plot PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Initial LR Testing\n2-class Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
plt.show()
```

Again, we see that metrics like area under ROC curve and accuracy (above) give a too optimistic impression of model performance. To reflect the unbalanced character of the credit card transaction data (0.1727% of the transactions were anomalous and 99.8273‬% were not anomalous), we also plotted the Precision-Recall curve. The area under the Precision-Recall curve (AUPRC) is a useful metric: if we had to assign this model a grade, 78/100 would be it. Not bad, but still much unexploited potential.

### LR hyperparameter tuning

To improve the performance of the Logistic Regression classifier, we will calibrate its main parameter C with a random grid search. The C-parameter fixes the inverse of regularization strength, and setting this parameter to a smaller value will increase the regularization strength.


```python
# Define hyperparameter grid
c_space = np.logspace(-3, 2, 51)
rand_grid = {'C': c_space,
             'solver': ['lbfgs'] }
print(rand_grid)

# Instantiate search object (use all cores but one)
grid = RandomizedSearchCV(LogisticRegression(random_state=21, max_iter=100), rand_grid,
                          n_iter = 20, cv=2, random_state=21, n_jobs = -2, verbose = 2)

# Fit object to data
grid.fit(X_train, y_train)

# Extract best model
optimized_clf = grid.best_estimator_
```


```python
# Print the tuned parameters and score
print('{:=^80}'.format('LR parameters for best candidate'))
print("Optimized Parameters: {}".format(grid.best_params_)) 
print("All Parameters: {}".format(optimized_clf.get_params())) 
print("Best score is {}".format(grid.best_score_))
```


```python
# Compute training metrics
accuracy = optimized_clf.score(X_train, y_train)

#  Predict labels of test set
train_pred = optimized_clf.predict(X_train)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_train, train_pred)
conf_mat = confusion_matrix(y_train.round(), train_pred.round())
clas_rep = classification_report(y_train.round(), train_pred.round())

# Print reports
print('{:=^80}'.format('Optimized LR training report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute testing metrics
accuracy = optimized_clf.score(X_test, y_test)

# Predict labels of test set
y_pred = optimized_clf.predict(X_test)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_test, y_pred)
conf_mat = confusion_matrix(y_test.round(), y_pred.round())
clas_rep = classification_report(y_test.round(), y_pred.round())

# Print reports
print('{:=^80}'.format('Optimized LR testing report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute predicted probabilities
y_pred_prob = optimized_clf.predict_proba(X_test)[:,1]

# Calculate receiver operating characteristics (ROC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k-')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Optimized LR Testing\nROC curve')
plt.show()
```


```python
# Compute AUPRC score
average_precision = average_precision_score(y_test, y_pred_prob)
print("AUPRC: {}".format(average_precision))

# Plot PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Optimized LR Testing\n2-class Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
plt.show()
```

*Logistic Regression final performance (above):* The model performance indicated by the AUPRC metric (area under Precision-Recall curve) did not improve substantially in comparison to the initial model, so it is time to try a different classifier.

## Kernel Support Vector Machine classification (SVC) <a id='three'></a>

Next, we will fit a kernel-type support vector classifier (SVC) with Gaussian radial basis function (RBF). Where logistic regression uses the output of a linear model, the SVC will define a hyperplane within the N-dimensional parameter space to classify the data points into either of the two categories--1 for anomalous transactions and 0 for all other transactions. This parameter space consists of the set of N predictor or feature variables. We will follow the same step-wise approach as for LR, starting with an initial model followed by hyperparameter tuning. 


```python
# Import modules
from sklearn.svm import SVC
```


```python
# Instantiate classifier (turning off probability increases speed)
clf = SVC(probability=True, gamma='scale', max_iter=-1, random_state=21)

# Fit classifier to training set
clf.fit(X_train, y_train)
```


```python
# Compute training metrics
accuracy = clf.score(X_train, y_train)

#  Predict labels of test set
train_pred = clf.predict(X_train)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_train, train_pred)
conf_mat = confusion_matrix(y_train.round(), train_pred.round())
clas_rep = classification_report(y_train.round(), train_pred.round())

# Print reports
print('{:=^80}'.format('Initial SVC training report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute testing metrics
accuracy = clf.score(X_test, y_test)

# Predict labels of test set
y_pred = clf.predict(X_test)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_test, y_pred)
conf_mat = confusion_matrix(y_test.round(), y_pred.round())
clas_rep = classification_report(y_test.round(), y_pred.round())

# Print reports
print('{:=^80}'.format('Initial SVC testing report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute predicted probabilities
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Calculate receiver operating characteristics (ROC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k-')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Initial SVC Testing\nROC curve')
plt.show()
```


```python
# Compute AUPRC score
average_precision = average_precision_score(y_test, y_pred_prob)
print("AUPRC: {}".format(average_precision))

# Plot PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Initial SVC Testing\n2-class Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
plt.show()
```

The Kernel SVC (above) performs better than LR because it predicts more true positives (anomalies recall=66%) and less false positives (anomalies precision=94%). At 81%, the SVC AUPRC, main metric of interest, is also greater than for LR.

### Kernel SVC hyperparameter tuning

To optimize the Kernel SVC, we will tune parameters C and gamma. Gamma defines the nonlinear hyperplane of the SVC, and represents the inverse of the radius of influence of samples identified by the model as support vectors. Because gamma determines how closely the hyperplane fits the training set, it follows that high values of gamma can lead to overfitting. Therefore, we use a moderate range. Additionally, we limit the number of iterations in the event that the classifier does not converge toward a solution.


```python
# Define hyperparameter grid
c_space = np.logspace(-3, 2, 51)
gamma_space = np.logspace(-3, 2, 51)
rand_grid = {'C': c_space,
             'gamma': gamma_space}
print(rand_grid)
```


```python
# Instantiate RandomizedSearchCV object (use all cores but one)
grid = RandomizedSearchCV(SVC(probability=True, random_state=21, max_iter = 1000), rand_grid,
                          n_iter = 20, cv=2, random_state=21, n_jobs = -2, verbose = 2)

# Fit object to data
grid.fit(X_train, y_train)

# Extract best model
optimized_clf = grid.best_estimator_
```


```python
# Print the tuned parameters and score
print('{:=^80}'.format('SVC parameters for best candidate'))
print("Optimized Parameters: {}".format(grid.best_params_)) 
print("All Parameters: {}".format(optimized_clf.get_params())) 
print("Best score is {}".format(grid.best_score_))
```


```python
# Compute training metrics
accuracy = optimized_clf.score(X_train, y_train)

#  Predict labels of test set
train_pred = optimized_clf.predict(X_train)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_train, train_pred)
conf_mat = confusion_matrix(y_train.round(), train_pred.round())
clas_rep = classification_report(y_train.round(), train_pred.round())

# Print reports
print('{:=^80}'.format('Optimized SVC training report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute testing metrics
accuracy = optimized_clf.score(X_test, y_test)

# Predict labels of test set
y_pred = optimized_clf.predict(X_test)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_test, y_pred)
conf_mat = confusion_matrix(y_test.round(), y_pred.round())
clas_rep = classification_report(y_test.round(), y_pred.round())

# Print reports
print('{:=^80}'.format('Optimized SVC testing report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute predicted probabilities
y_pred_prob = optimized_clf.predict_proba(X_test)[:,1]

# Calculate receiver operating characteristics (ROC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k-')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Optimized SVC Testing\nROC curve')
plt.show()
```


```python
# Compute AUPRC score
average_precision = average_precision_score(y_test, y_pred_prob)
print("AUPRC: {}".format(average_precision))

# Plot PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Optimized SVC Testing\n2-class Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
plt.show()
```

*Kernel Support Vector Classifier final performance (above):* 

* Hyperparameter tuning the Kernel SVC did not result in a better classifier than the Kernel SVC we started with. 
* Some Kernel SVC parameter combination failed to converge within the maximum number of iterations specified. 
* Regardless, Kernel SVC performed notably better than the LR in terms of precision (low number of false negatives and false positives), recall and area under Precision-Recall curve (AUPRC). 

At this point we could decide to try again and explore the parameter space in more detail, but let's try other classifiers instead. 

## Gradient Boosting Model (GBM) classification <a id='four'></a>

Gradient Boosting builds an additive model in a forward step-wise approach, by fitting a single regression tree (in binary classification) that optimizes the deviance loss function. As such, a GBM combines both parametric and non-parametric methods.


```python
# Import modules
from sklearn.ensemble import GradientBoostingClassifier
```


```python
# Instantiate classifier
clf = GradientBoostingClassifier(random_state=21, verbose=1)

# Fit classifier to the training set
clf.fit(X_train, y_train)
```


```python
# Compute training metrics
accuracy = clf.score(X_train, y_train)

#  Predict labels of test set
train_pred = clf.predict(X_train)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_train, train_pred)
conf_mat = confusion_matrix(y_train.round(), train_pred.round())
clas_rep = classification_report(y_train.round(), train_pred.round())

# Print reports
print('{:=^80}'.format('Initial SVC training report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute testing metrics
accuracy = clf.score(X_test, y_test)

# Predict labels of test set
y_pred = clf.predict(X_test)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_test, y_pred)
conf_mat = confusion_matrix(y_test.round(), y_pred.round())
clas_rep = classification_report(y_test.round(), y_pred.round())

# Print reports
print('{:=^80}'.format('Initial GBM testing report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute predicted probabilities
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Calculate receiver operating characteristics (ROC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k-')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Initial GBM Testing\nROC curve')
plt.show()
```


```python
# Compute AUPRC score
average_precision = average_precision_score(y_test, y_pred_prob)
print("AUPRC: {}".format(average_precision))

# Plot PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Initial GBM Testing\n2-class Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
plt.show()
```

### GBM hyperparameter tuning


```python
# Define hyperparameter grid
learning_rate = [0.02, 0.1]
n_estimators = [int(x) for x in [100, 200, 300, 400]]
subsample = [0.5, 0.9]
max_depth = [int(x) for x in [3, 4, 5, 10]]
min_samples_split = [int(x) for x in [2, 3, 4, 5]]
rand_grid = {'learning_rate': learning_rate,
             'n_estimators': n_estimators,
             'subsample': subsample,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split}
print(rand_grid)
```


```python
# Instantiate RandomizedSearchCV object (use all cores but one)
grid = RandomizedSearchCV(GradientBoostingClassifier(validation_fraction=0.3, n_iter_no_change=10, random_state=21), rand_grid, 
                          n_iter = 20, cv=2, random_state=21, n_jobs = -2, verbose = 2)

# Fit object to data
grid.fit(X_train, y_train)

# Extract best model
optimized_clf = grid.best_estimator_
```


```python
# Print the tuned parameters and score
print('{:=^80}'.format('GBM parameters for best candidate'))
print("Optimized Parameters: {}".format(grid.best_params_)) 
print("All Parameters: {}".format(optimized_clf.get_params())) 
print("Best score is {}".format(grid.best_score_))
```


```python
# Compute training metrics
accuracy = optimized_clf.score(X_train, y_train)

#  Predict labels of test set
train_pred = optimized_clf.predict(X_train)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_train, train_pred)
conf_mat = confusion_matrix(y_train.round(), train_pred.round())
clas_rep = classification_report(y_train.round(), train_pred.round())

# Print reports
print('{:=^80}'.format('Optimized GBM training report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute testing metrics
accuracy = optimized_clf.score(X_test, y_test)

# Predict labels of test set
y_pred = optimized_clf.predict(X_test)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_test, y_pred)
conf_mat = confusion_matrix(y_test.round(), y_pred.round())
clas_rep = classification_report(y_test.round(), y_pred.round())

# Print reports
print('{:=^80}'.format('Optimized GBM testing report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute predicted probabilities
y_pred_prob = optimized_clf.predict_proba(X_test)[:,1]

# Calculate receiver operating characteristics (ROC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k-')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Optimized GBM Testing\nROC curve')
plt.show()
```


```python
# Compute AUPRC score
average_precision = average_precision_score(y_test, y_pred_prob)
print("AUPRC: {}".format(average_precision))

# Plot PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Optimized GBM Testing\n2-class Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
plt.show()
```

With an AUPRC of 0.7164 (above), the gradient boosting model performed worse than both the LR classifier and Kernel SVC. Lower performance is mostly explained by the lower recall for anomalies (value 1). While the GBM was on point for the anomalies it identified, it also missed a large percentage of anomalies--at least more than the LR classifier and Kernel SVC.  

## Random Forest (RF) classification <a id='five'></a>

Until now we used LR, Kernel SVC and GBM to make predictions, and all three have limitations when it comes to classification of unbalanced data. LR is a special case of generalized linear model (GLM) and makes assumptions about the underlying data distribution. This method works best with uncorrelated data and logarithmic error distributions, and therefore requires many data samples for fitting. Kernel SVC fitting involves the computation of polynomial surfaces, and the nonlinear nature of polynomial calculations makes that Kernel SVCs are not easily parallelized. GBM builds trees one at a time, each attempting to explain the residual error of the previous tree. While this can work for balanced datasets, the sequential nature of this process can be a disadvantage. 

Random Forest (RF) classification offers a non-parametric approach where a large number of uncorrelated models (decision trees) are fitted to the data, and vote independently as a joint committee on what the outcome of each prediction should be. One advantage of RF is that it makes no assumptions about the sample distribution or error distribution, meaning the classifier is robust and not biased by outliers. Furthermore, RF can be parallelized into as many processes as there are estimators (trees). Because a RF calculates fast, we will not hypertune the parameters but instead allow it to fit the data without constraints. 


```python
# Import modules
from sklearn.ensemble import RandomForestClassifier
```


```python
# Fit random forest for range of maximum depth of tree 
max_depths = [int(x) for x in [2,4,8,16,32,64]]

train_results = []
test_results = []
for max_depth in max_depths:
   clf = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=21, n_jobs = -2)
   clf.fit(X_train, y_train)
   train_pred_prob = clf.predict_proba(X_train)[:,1]
   average_precision = average_precision_score(y_train, train_pred_prob)
   train_results.append(average_precision)
   y_pred_prob = clf.predict_proba(X_test)[:,1]
   average_precision = average_precision_score(y_test, y_pred_prob)
   test_results.append(average_precision)
```


```python
# Plot AUPRC vs tree depth
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Train AUPRC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUPRC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUPRC score')
plt.xlabel('Tree depth')
plt.title('Area under Precision-Recall curve vs Tree Depth')
plt.show()
```

The maximum area under the Precision-Recall curve (AUPRC) is reached for a tree depth less than 20. It appears that an AUPRC of ~0.84 is the best possible testing performance we may expect given a training AUPRC of close to 1.00.

### Random Forest without constraints


```python
# Import modules
from sklearn.ensemble import RandomForestClassifier
```


```python
# Zero ratios
y_train_s = np.prod(y_train.shape)
y_train_z = (y_train_s - np.sum(y_train)) / y_train_s
y_test_s = np.prod(y_test.shape)
y_test_z = (y_test_s - np.sum(y_test)) / y_test_s
print("Zero ratio in training labels: {}".format(y_train_z))
print("Zero ratio in testing labels: {}".format(y_test_z))
```


```python
# Compute sample weights for unbalanced classes as inverse of probability
weight_0 = 1.0
weight_1 = (1 - y_train_z)**-1
sample_weight = np.array([weight_1 if i == 1 else weight_0 for i in enumerate(y_train)])
print("Sample weight for logical(0): {}".format(weight_0))
print("Sample weight for logical(1): {}".format(weight_1))
```


```python
# Instantiate classifier
clf = RandomForestClassifier(n_estimators=200, random_state=21, n_jobs = -2, verbose = 2)

# Fit classifier to training set
clf = clf.fit(X_train, y_train, sample_weight=sample_weight)
```


```python
# Compute training metrics
accuracy = clf.score(X_train, y_train)

#  Predict labels of test set
train_pred = clf.predict(X_train)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_train, train_pred)
conf_mat = confusion_matrix(y_train.round(), train_pred.round())
clas_rep = classification_report(y_train.round(), train_pred.round())

# Print reports
print('{:=^80}'.format('RF training report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute testing metrics
accuracy = clf.score(X_test, y_test)

# Predict labels of test set
y_pred = clf.predict(X_test)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_test, y_pred)
conf_mat = confusion_matrix(y_test.round(), y_pred.round())
clas_rep = classification_report(y_test.round(), y_pred.round())

# Print reports
print('{:=^80}'.format('RF testing report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute predicted probabilities
y_pred_prob = clf.predict_proba(X_test)[:,1]
```


```python
# Calculate receiver operating characteristics (ROC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k-')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('RF Testing\nROC curve')
plt.show()
```


```python
# Compute AUPRC score
average_precision = average_precision_score(y_test, y_pred_prob)
print("AUPRC: {}".format(average_precision))

# Plot PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('RF Testing\n2-class Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
plt.show()
```

## Model selection and cost-effective optimization <a id='six'></a>

*Selecting the best classifier:* We have tested four classifiers: Logistic Regression, Kernel Support Vector Classifier, Stochastic Gradient Boosting and Random Forest. Which one is the best choice? A practical way to evaluate the suitability of a classifier for daily use is in terms of cost-effectiveness: we want the classifier to be able to make predictions on new data (high testing precision and recall), and at a low cost (fast computation and modest data requirements). 

The Random Forest (AUPRC=0.8456) performed best in terms of precision and recall, followed by Kernel Support Vector Classifier (AUPRC=0.8081), Logistic Regression (AUPRC=0.7822) and Stochastic Gradient Boosting Machine (AUPRC=0.7164). RF and LR computed fastest. Conversely, Kernel SVC used the most computer time. While in a real-world scenario we might be able to obtain a better model (probably SGB or RF) given enough training data and more iterations, we assume that RF provides the most cost-effective prediction of transaction anomalies for now. 

### Feature importance

Now that we have identified the RF as the most suitable model to predict transaction anomalies, the next step is to see if can improve its cost-effectiveness by reducing data requirements. We trained the model on 29 features, so let's find out which of those features contribute the most information to the prediction. 


```python
# Get feature importances
feature_list = list(df.columns[:-1])
importances = list(clf.feature_importances_)
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print feature ranking
print("Feature ranking:")
_ = [print('Variable: {:3} Importance: {}'.format(*pair)) for pair in feature_importances]
```


```python
# Plot feature ranking in bar chart
X_values = list(range(len(importances)))
plt.bar(X_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
plt.xticks(X_values, feature_list, rotation = 'vertical')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Feature Relative Importance')
plt.show()
```


```python
# List of features sorted by decreasing importance
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]

# Cumulative importance
cumulative_importances = np.cumsum(sorted_importances)

# Create line plot
plt.plot(X_values, cumulative_importances, 'b-')
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
plt.xticks(X_values, sorted_features, rotation = 'vertical')
plt.xlabel('Feature')
plt.ylabel('Cumulative Importance')
plt.title('Feature Cumulative Importance')
plt.show()
```


```python
# Number of features explaining 95% cum. importance
n_import = np.where(cumulative_importances > 0.95)[0][0] + 1
print('Number of features required (95% importance):', n_import)

# Least important features
limp_feature_names = sorted_features[-(len(importances)-n_import):]
print('Least important features (5% importance):', limp_feature_names)
```

Feature importance analysis (above) shows that 24 features out of the available set of 29 features account for 95% of the Gini Importance. There are three things we learn from this: 

1. There is a relative lack of correlation between features, confirming that these are indeed orthogonally transformed data (in this case the outcome of principal component analysis); 
2. The RF classifier makes optimum use of the training data; 
3. We can drop 5 of the features without incurring a high cost to model performance, namely the PCAs labeled as 28, 13, 24, 25 and 23. 

### Retrain classifier on the most important features


```python
# Extract the names of most important features
important_feature_names = [feature[0] for feature in feature_importances[0:(n_import - 1)]]

# Find the columns of the most important features
important_indices = [feature_list.index(feature) for feature in important_feature_names]

# Create training and testing sets with only important features
X_train_imp = X_train[:,important_indices]
X_test_imp = X_test[:,important_indices]

# Print dimensions
print("Dimensions of X_train_imp: {}".format(X_train_imp.shape))
print("Dimensions of y_train_imp: {}".format(y_train.shape))
print("Dimensions of X_test_imp: {}".format(X_test_imp.shape))
print("Dimensions of y_test_imp: {}".format(y_test.shape))
```


```python
# Fit classifier to training set
clf = clf.fit(X_train_imp, y_train, sample_weight=sample_weight)
```


```python
# Compute training metrics
accuracy = clf.score(X_train_imp, y_train)

#  Predict labels of test set
train_pred = clf.predict(X_train_imp)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_train, train_pred)
conf_mat = confusion_matrix(y_train.round(), train_pred.round())
clas_rep = classification_report(y_train.round(), train_pred.round())

# Print reports
print('{:=^80}'.format('New RF training report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute testing metrics
accuracy = clf.score(X_test_imp, y_test)

# Predict labels of test set
y_pred = clf.predict(X_test_imp)

# Compute MSE, confusion matrix, classification report
mse = mean_squared_error(y_test, y_pred)
conf_mat = confusion_matrix(y_test.round(), y_pred.round())
clas_rep = classification_report(y_test.round(), y_pred.round())

# Print reports
print('{:=^80}'.format('New RF testing report'))
print('Accuracy: %.4f' % accuracy)
print("MSE: %.4f" % mse)
print("Confusion matrix:\n{}".format(conf_mat))
print("Classification report:\n{}".format(clas_rep))
```


```python
# Compute predicted probabilities
y_pred_prob = clf.predict_proba(X_test_imp)[:,1]
```


```python
# Compute AUPRC score
average_precision = average_precision_score(y_test, y_pred_prob)
print("AUPRC: {}".format(average_precision))

# Plot PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('New RF Testing\n2-class Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
plt.show()
```

The testing AUPRC has not decreased after retraining the RF with the 24 most important of 29 features. We now have a more cost-effective RF classifier that has the same predictive power, but requires less data to train. These are the final scores:  
 
<table>
  <tr>
    <th>Classifier</th>
    <th>AUPRC</th>
  </tr>
  <tr>
    <td>Random Forest (24 most important features)</td>
    <td>0.8487</td>
  </tr>
  <tr>
    <td>Random Forest (all 29 features)</td>
    <td>0.8456</td>
  </tr>
  <tr>
    <td>Kernel Support Vector Classifier</td>
    <td>0.8081</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>0.7822</td>
  </tr>
  <tr>
    <td>Stochastic Boosted Regression</td>
    <td>0.7164</td>
  </tr>
</table> 

## Conclusion <a id='seven'></a> 

* Conventional metrics for classification models like accuracy and mean squared error give a too optimistic impression of model performance. Because the transaction dataset is a highly unbalanced dataset (anomalous transactions accounted for 0.1727% of all transactions), the area under Precision-Recall curve (AUPRC) is a more useful metric of model performance. 

* The Random Forest provided the most cost-effective anomaly detection in terms of precision and recall (AUPRC=0.8487) and computation time, followed by the fast Logistic Regression (AUPRC=0.7822) and slower Kernel Support Vector Classifier (AUPRC=0.8081) and Stochastic Gradient Boosting Machine (AUPRC=0.7164). 

* 5 the original 29 features can be dropped without significant cost to model performance.  
