"""
GROUP 8 - MACHINE LEARNING WITH PYTHON ASSIGNMENTS 4 & 5

Alexander Lund, Célian Marx, Augustin Tixier

date: 09.04.2021
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

################## ASSIGNMENT 4 ##################


train = pd.read_csv("../../../PycharmProjects/FundPySciESCP/ML with Python/train.csv")


# change -1 to 0s for binary simplicity
for x in range(len(train["label"])):
    if train["label"][x] == -1:
        train["label"][x] = 0
        train["purchaseTime"][x] = 0

"""
We just verify that the timestamp corresponds to an actual hour of the day, regardless of which date (since pd.to_datetime returns with starting point 1970-01-01)

# convert timestamp to datetime for both visitTime and purchaseTime
train["visitTime"] = pd.to_datetime(train["visitTime"], unit="s")
train["purchaseTime"] = pd.to_datetime(train["purchaseTime"], unit="s")

"""

# check how imbalanced the predict column is
print(train["label"].value_counts()) # we see that we only have 57 "1"s, against 31'371 "-1"s

# convert hashed values into categorical values for the model
train[["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"]] = train[["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"]].astype("category")

# check heatmap for collinearity using seaborn (we will verify multicolinearity using Variance Inflation Factor (VIF) below)
traincorr = train.corr()
plt.figure(figsize=(40,35))
heat = sns.heatmap(traincorr, annot=True)
plt.show()

# we create a VIF dataframe and use variance inflation factor package to detect the variables with the highest multicolinearity
# the VIF are calculated by taking the predictor value, and regress it against every other prediction variables to determine which values have highest multicolinearity
vif_data = pd.DataFrame()
vif_data["feature"] = train.columns

vif_data["VIF"] = [VIF(train.values, i) for i in range(len(train.columns))]

print(vif_data)

# in general VIF results around 1 are not correlated, between 1 and 5 are moderately correlated and above 5 are highly correlated
# in our case, we have "visitTime", "purchaseTime" "N8", "N9" and "N10" which have a VIF result above 5 and we therefore drop those
train = train.drop(columns=["visitTime", "purchaseTime", "N8", "N9", "N10"])

# split into dependent and independent
Xtrain = train.drop(columns="label")
ytrain = train["label"]


# load test dataset
test = pd.read_csv("../../../PycharmProjects/FundPySciESCP/ML with Python/test.csv")

# change -1 to 0s for binary simplicity
for x in range(len(test["label"])):
    if test["label"][x] == -1:
        test["label"][x] = 0
        test["purchaseTime"][x] = 0

# categorize hashed values for test dataset as with train dataset
test[["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"]] = test[["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"]].astype("category")

# drop same multicolinear columns as determined with VIF for train dataset
test = test.drop(columns=["visitTime", "purchaseTime", "N8", "N9", "N10"])

#split into dependent and independent values
Xtest = test.drop(columns="label")
ytest = test["label"]

print("\n")

### Logisitic Regression ###
print("Logistic Regression:")
# we use Logistic Regression from sklearn with a Ridge penalty (L2) as not to penalize too much the imbalanced data and we use newton-cg for solver to minimise quadratic function (uses quadratic approximation, first & second partial derivatives)
model = LogisticRegression(penalty="l2", solver="newton-cg")

# we fit our trained data to the Logistic Regression and create a predicted Series (binary outcome)
model.fit(Xtrain, ytrain)
predict = model.predict(Xtest)

# we create a confusion matrix to score the model
matrix = confusion_matrix(ytest, predict, labels=[1,0])
print(matrix) # we notice that the test dataset is completely imbalanced, there are no positive values to predict

# the classification report indicates the precision, recall, f1-score and supports of the model
print(classification_report(ytest, predict))

# we also print the Pearson's coefficient
print(model.score(Xtest, ytest)) # we have the highest coefficient, since there all values are True Negatives


print("\n")

# prepare probabilities into separate submission .csv
y_proba = model.predict_proba(Xtest)

y_proba = pd.DataFrame(y_proba)

purchase_prob = y_proba[1]

purchase_prob = pd.DataFrame(purchase_prob)

submission = pd.concat([test["id"],purchase_prob], axis=1)

submission = submission.rename(columns={1:"probability_purchase"})

submission.to_csv("Final-Prob_Lund_Marx_Tixier.csv", index=False)


################## EXTRA: how to deal with imbalanced data (trial) ##################

# We use Synthetic Minority Oversampling (SMOTE) to extrapolate and determine new 1s in label
# SMOTE uses a nearest neighbor method to determine additional positive values in predict column
# this enables to rebalance dataset in order to not only have True Negatives as with original test dataset
X_resampled, y_resampled = SMOTE().fit_resample(Xtrain, ytrain)

# split resampled (SMOTE) data into train & test
Xtr_resampled, Xte_resampled, ytr_resampled, yte_resampled = train_test_split(X_resampled, y_resampled, test_size=0.3)

# fit splitted resampled data to Logistic Regression
LOGSMOTE = model.fit(Xtr_resampled, ytr_resampled)
predict2 = model.predict(Xte_resampled)

# create confusion matrix
matrix2 = confusion_matrix(yte_resampled, predict2, labels=[1,0])
print(matrix2) # we see that we have 9339 TPs (15 FPs), and 9469 TNs

# the model still has a very high precision, recall and accuracy
print(classification_report(yte_resampled, predict2))

# the ROC curve is almost perfect (flat along both axis)
plot_roc_curve(LOGSMOTE, Xte_resampled, yte_resampled)
plt.show()

# trying to determine probabilities with SMOTE (high degree of NaN values in ID due to SMOTE)
ySMOTE_proba = model.predict_proba(Xte_resampled)

ySMOTE_proba = pd.DataFrame(ySMOTE_proba)

purchase_probSMOTE = ySMOTE_proba[1]

purchase_probSMOTE = pd.DataFrame(purchase_probSMOTE)

SMOTEprob = pd.concat([Xte_resampled["id"],purchase_probSMOTE], axis=1)

SMOTEprob = SMOTEprob.rename(columns={1:"probability_purchase"})

"""print(SMOTEprob)"""

print("\n")
"""
We tried using SMOTE in order to rebalance the dataset, however we had issues with the generation and addition 
of new NaN values in almost every column. We think that this could be improved further but we did not want to
drop the NaN values and our trial at rebalancing the dataset proved that we could have an almost perfect split
in True Positives against True Negatives (in comparison to just using Logisitic Regression and therefore having
only True Negatives.
"""