# Breast Cancer Prediction
##  Importing libraries
# Importing libraries

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

import warnings

warnings.filterwarnings('ignore')

sns.set()
plt.style.use('ggplot')
## Load the data
df = pd.read_csv('../data/cancer.csv')
df.head()
## Data Preprocessing
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df.diagnosis.unique()
df['diagnosis'] = df['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)
df.head()
df.describe()
df.info()
# checking for null values

df.isna().sum()
# visualizing null values

msno.bar(df)
#### There are no missing values in the data.
df.describe()
plt.hist(df['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()
## Exploratory Data Analysis (EDA)
plt.figure(figsize=(20, 15))
plotnumber = 1

for column in df:
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column)

    plotnumber += 1

plt.tight_layout()
plt.show()
# heatmap

plt.figure(figsize=(20, 12))

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, linewidths=1, annot=True, fmt=".2f")
plt.show()
### We can see that there are many columns which are very highly correlated which causes multicollinearity so we have to remove highly correlated features.
# removing highly correlated features

corr_matrix = df.corr().abs()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.92)]

df = df.drop(to_drop, axis=1)

print(f"The reduced dataframe has {df.shape[1]} columns.")
to_drop
df.info()
# creating features and label

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
# splitting data into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
# scaling data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Logistic Regression
# fitting data to model

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# model predictions

y_pred = log_reg.predict(X_test)
# accuracy score

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(accuracy_score(y_train, log_reg.predict(X_train)))

log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
print(log_reg_acc)
# confusion matrix

print(confusion_matrix(y_test, y_pred))
# classification report

print(classification_report(y_test, y_pred))
# K Neighbors Classifier (KNN)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# model predictions

y_pred = knn.predict(X_test)
# accuracy score

print(accuracy_score(y_train, knn.predict(X_train)))

knn_acc = accuracy_score(y_test, knn.predict(X_test))
print(knn_acc)
# confusion matrix

print(confusion_matrix(y_test, y_pred))
# classification report

print(classification_report(y_test, y_pred))
# Support Vector Machine (SVM)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC(probability=True)
parameters = {
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'C': [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svc, parameters)
grid_search.fit(X_train, y_train)
# best parameters

grid_search.best_params_
# best score

grid_search.best_score_
svc = SVC(C=10, gamma=0.01, probability=True)
svc.fit(X_train, y_train)
# model predictions

y_pred = svc.predict(X_test)
# accuracy score

print(accuracy_score(y_train, svc.predict(X_train)))

svc_acc = accuracy_score(y_test, svc.predict(X_test))
print(svc_acc)
# confusion matrix

print(confusion_matrix(y_test, y_pred))
# classification report

print(classification_report(y_test, y_pred))
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 32, 1),
    'min_samples_leaf': range(1, 10, 1),
    'min_samples_split': range(2, 10, 1),
    'splitter': ['best', 'random']
}

grid_search_dt = GridSearchCV(dtc, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_dt.fit(X_train, y_train)
# best parameters

grid_search_dt.best_params_
# best score

grid_search_dt.best_score_
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=19, min_samples_leaf=4, min_samples_split=6,
                             splitter='random')
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
# accuracy score

print(accuracy_score(y_train, dtc.predict(X_train)))

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))
print(dtc_acc)
# confusion matrix

print(confusion_matrix(y_test, y_pred))
# classification report

print(classification_report(y_test, y_pred))
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='auto', min_samples_leaf=2,
                                  min_samples_split=3, n_estimators=130)
rand_clf.fit(X_train, y_train)
y_pred = rand_clf.predict(X_test)
# accuracy score

print(accuracy_score(y_train, rand_clf.predict(X_train)))

ran_clf_acc = accuracy_score(y_test, y_pred)
print(ran_clf_acc)
# confusion matrix

print(confusion_matrix(y_test, y_pred))
# classification report

print(classification_report(y_test, y_pred))
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.1],
    'n_estimators': [100, 150, 180]
}

grid_search_gbc = GridSearchCV(gbc, parameters, cv=2, n_jobs=-5, verbose=1)
grid_search_gbc.fit(X_train, y_train)
# best parameters

grid_search_gbc.best_params_
# best score

grid_search_gbc.best_score_
gbc = GradientBoostingClassifier(learning_rate=0.1, loss='exponential', n_estimators=180)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
# accuracy score

print(accuracy_score(y_train, gbc.predict(X_train)))

gbc_acc = accuracy_score(y_test, y_pred)
print(gbc_acc)
# confusion matrix

print(confusion_matrix(y_test, y_pred))
# classification report

print(classification_report(y_test, y_pred))
# Extreme Gradient Boosting
from xgboost import XGBClassifier

xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.01, max_depth=5, n_estimators=180)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
# accuracy score

print(accuracy_score(y_train, xgb.predict(X_train)))

xgb_acc = accuracy_score(y_test, y_pred)
print(xgb_acc)
# confusion matrix

print(confusion_matrix(y_test, y_pred))
# classification report

print(classification_report(y_test, y_pred))
# Model Comparison
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'SVM', 'Decision Tree Classifier', 'Random Forest Classifier',
              'Gradient Boosting Classifier', 'XgBoost'],
    'Score': [100 * round(log_reg_acc, 4), 100 * round(knn_acc, 4), 100 * round(svc_acc, 4), 100 * round(dtc_acc, 4),
              100 * round(ran_clf_acc, 4),
              100 * round(gbc_acc, 4), 100 * round(xgb_acc, 4)]
})
models.sort_values(by='Score', ascending=False)
import pickle

model = svc
pickle.dump(model, open("breast_cancer.pkl", 'wb'))
from sklearn import metrics

plt.figure(figsize=(8, 5))
models = [
    {
        'label': 'LR',
        'model': log_reg,
    },
    {
        'label': 'DT',
        'model': dtc,
    },
    {
        'label': 'SVM',
        'model': svc,
    },
    {
        'label': 'KNN',
        'model': knn,
    },
    {
        'label': 'XGBoost',
        'model': xgb,
    },
    {
        'label': 'RF',
        'model': rand_clf,
    },
    {
        'label': 'GBDT',
        'model': gbc,
    }
]
for m in models:
    model = m['model']
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fpr1, tpr1, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc = metrics.roc_auc_score(y_test, model.predict(X_test))
    plt.plot(fpr1, tpr1, label='%s - ROC (area = %0.2f)' % (m['label'], auc))

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
plt.title('ROC - Breast Cancer Prediction', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
# plt.savefig("outputs/roc_breast_cancer.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show()
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

models = [
    {
        'label': 'LR',
        'model': log_reg,
    },
    {
        'label': 'DT',
        'model': dtc,
    },
    {
        'label': 'SVM',
        'model': svc,
    },
    {
        'label': 'KNN',
        'model': knn,
    },
    {
        'label': 'XGBoost',
        'model': xgb,
    },
    {
        'label': 'RF',
        'model': rand_clf,
    },
    {
        'label': 'GBDT',
        'model': gbc,
    }
]

means_roc = []
means_accuracy = [100 * round(log_reg_acc, 4), 100 * round(dtc_acc, 4), 100 * round(svc_acc, 4),
                  100 * round(knn_acc, 4), 100 * round(xgb_acc, 4),
                  100 * round(ran_clf_acc, 4), 100 * round(gbc_acc, 4)]

for m in models:
    model = m['model']
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fpr1, tpr1, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc = metrics.roc_auc_score(y_test, model.predict(X_test))
    auc = 100 * round(auc, 4)
    means_roc.append(auc)

print(means_accuracy)
print(means_roc)

# data to plot
n_groups = 7
means_accuracy = tuple(means_accuracy)
means_roc = tuple(means_roc)

# create plot
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_accuracy, bar_width,
                 alpha=opacity,
                 color='mediumpurple',
                 label='Accuracy (%)')

rects2 = plt.bar(index + bar_width, means_roc, bar_width,
                 alpha=opacity,
                 color='rebeccapurple',
                 label='ROC (%)')

plt.xlim([-1, 8])
plt.ylim([70, 104])

plt.title('Performance Evaluation - Breast Cancer Prediction', fontsize=12)
plt.xticks(index, ('   LR', '   DT', '   SVM', '   KNN', 'XGBoost', '   RF', '   GBDT'), rotation=40, ha='center',
           fontsize=12)
plt.legend(loc="upper right", fontsize=10)
# plt.savefig("outputs/PE_breast_cancer.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show()
