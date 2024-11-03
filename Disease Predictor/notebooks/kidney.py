# Kidney Disease Prediction
# necessary imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
sns.set()
plt.style.use('ggplot')
% matplotlib
inline
pd.set_option('display.max_columns', 26)
# loading data

df = pd.read_csv('../data/kidney.csv')
df.head()
df.shape
# dropping id column
df.drop('id', axis=1, inplace=True)
# rename column names to make it more user-friendly

df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']
df.head()
df.describe()
df.info()
# As we can see that 'packed_cell_volume', 'white_blood_cell_count' and 'red_blood_cell_count' are object type. We need to change them to numerical dtype.
# converting necessary columns to numerical type

df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

df.info()
# Extracting categorical and numerical columns

cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']
# looking at unique values in categorical columns

for col in cat_cols:
    print(f"{col} has {df[col].unique()} values\n")
# There is some ambugity present in the columns we have to remove that.

# replace incorrect values

df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)

df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace='\tno', value='no')

df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})
df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
df['class'] = pd.to_numeric(df['class'], errors='coerce')
cols = ['diabetes_mellitus', 'coronary_artery_disease', 'class']

for col in cols:
    print(f"{col} has {df[col].unique()} values\n")
# checking numerical features distribution

plt.figure(figsize=(20, 15))
plotnumber = 1

for column in num_cols:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column)

    plotnumber += 1

plt.tight_layout()
plt.show()
# looking at categorical columns

# plt.figure(figsize = (20, 15))
# plotnumber = 1

# for column in cat_cols:
#     if plotnumber <= 11:
#         ax = plt.subplot(3, 4, plotnumber)
#         sns.countplot(df[column], palette = 'rocket')
#         plt.xlabel(column)

#     plotnumber += 1

# plt.tight_layout()
# plt.show()
# heatmap of data

plt.figure(figsize=(15, 8))

sns.heatmap(df.corr(), annot=True, linewidths=2, linecolor='lightgrey')
plt.show()
df.columns


# EXPLORATY DATA ANALYSIS (EDA)
# defining functions to create plot

def violin(col):
    fig = px.violin(df, y=col, x="class", color="class", box=True, template='plotly_dark')
    return fig.show()


def kde(col):
    grid = sns.FacetGrid(df, hue="class", height=6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()


def scatter(col1, col2):
    fig = px.scatter(df, x=col1, y=col2, color="class", template='plotly_dark')
    return fig.show()


kde('red_blood_cell_count')
kde('white_blood_cell_count')

kde('packed_cell_volume')
kde('haemoglobin')
kde('albumin')
kde('blood_glucose_random')
kde('blood_urea')

kde('specific_gravity')
# DATA PRE PROCESSING
# checking for null values

df.isna().sum().sort_values(ascending=False)
df[num_cols].isnull().sum()
df[cat_cols].isnull().sum()


# filling null values, we will use two methods, random sampling for higher null values and
# mean/mode sampling for lower null values

def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample


def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)


# filling num_cols null values using random sampling method

for col in num_cols:
    random_value_imputation(col)
df[num_cols].isnull().sum()
# filling "red_blood_cells" and "pus_cell" using random sampling method and rest of cat_cols using mode imputation

random_value_imputation('red_blood_cells')
random_value_imputation('pus_cell')

for col in cat_cols:
    impute_mode(col)
df[cat_cols].isnull().sum()
# All the missing values are handeled now, lets do ctaegorical features encoding now

# FEATURE ENCODING
for col in cat_cols:
    print(f"{col} has {df[col].nunique()} categories\n")
# As all of the categorical columns have 2 categories we can use label encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])
df.head()
# MODEL BUILDING
ind_col = [col for col in df.columns if col != 'class']
dep_col = 'class'

X = df[ind_col]
y = df[dep_col]
# splitting data intp training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of knn

knn_acc = accuracy_score(y_test, knn.predict(X_test))

print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}")
print(f"Test Accuracy of KNN is {knn_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, knn.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, knn.predict(X_test))}")
# DECISION TREE
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of decision tree

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")
# hyper parameter tuning of decision tree

from sklearn.model_selection import GridSearchCV

grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10],
    'splitter': ['best', 'random'],
    'min_samples_leaf': [1, 2, 3, 5, 7],
    'min_samples_split': [1, 2, 3, 5, 7],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search_dtc = GridSearchCV(dtc, grid_param, cv=5, n_jobs=-1, verbose=1)
grid_search_dtc.fit(X_train, y_train)
# best parameters and best score

print(grid_search_dtc.best_params_)
print(grid_search_dtc.best_score_)
# best estimator

dtc = grid_search_dtc.best_estimator_

# accuracy score, confusion matrix and classification report of decision tree

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")
# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rd_clf = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='auto', min_samples_leaf=5,
                                min_samples_split=2, n_estimators=400)
rd_clf.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of random forest

rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))

print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, rd_clf.predict(X_train))}")
print(f"Test Accuracy of Random Forest Classifier is {rd_clf_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, rd_clf.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, rd_clf.predict(X_test))}")
# XgBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.001, max_depth=10, n_estimators=100)
xgb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of xgboost

xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

print(f"Training Accuracy of XgBoost is {accuracy_score(y_train, xgb.predict(X_train))}")
print(f"Test Accuracy of XgBoost is {xgb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, xgb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, xgb.predict(X_test))}")
# Logistic Regression
# fitting data to model

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
# model predictions

y_pred = lr.predict(X_test)
# accuracy score

print(accuracy_score(y_train, lr.predict(X_train)))

lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(lr_acc)
# confusion matrix

print(confusion_matrix(y_test, y_pred))
# classification report

print(classification_report(y_test, y_pred))
# SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC(probability=True)
parameters = {
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'C': [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svm, parameters)
grid_search.fit(X_train, y_train)
# best parameters

grid_search.best_params_
grid_search.best_score_
svm = SVC(C=10, gamma=0.001, probability=True)
svm.fit(X_train, y_train)
# model predictions

y_pred = svm.predict(X_test)
# accuracy score

print(accuracy_score(y_train, svm.predict(X_train)))

svm_acc = accuracy_score(y_test, svm.predict(X_test))
print(svm_acc)
# confusion matrix

print(confusion_matrix(y_test, y_pred))
# classification report

print(classification_report(y_test, y_pred))
# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.1, 1, 10],
    'n_estimators': [100, 150, 180, 200]
}

grid_search_gbc = GridSearchCV(gbc, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search_gbc.fit(X_train, y_train)
# best parameters

grid_search_gbc.best_params_
# best score

grid_search_gbc.best_score_
gbc = GradientBoostingClassifier(learning_rate=1, loss='deviance', n_estimators=150)
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
# MODELS COMPARISON
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'SVM', 'Decision Tree Classifier', 'Random Forest Classifier', 'XgBoost',
              'Gradient Boosting'],
    'Score': [lr_acc, knn_acc, svm_acc, dtc_acc, rd_clf_acc, xgb_acc, gbc_acc]
})

models.sort_values(by='Score', ascending=False)
import pickle

model = rd_clf
pickle.dump(model,
            open(r"C:\Users\ganes\OneDrive\Desktop\Medibuddy-Smart-Disease-Predictor-main\models\kidney.pkl", 'wb'))
from sklearn import metrics

plt.figure(figsize=(8, 5))
models = [
    {
        'label': 'LR',
        'model': lr,
    },
    {
        'label': 'DT',
        'model': dtc,
    },
    {
        'label': 'SVM',
        'model': svm,
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
        'model': rd_clf,
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
plt.title('ROC - Kidney Disease Prediction', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
# plt.savefig("outputs/roc_kidney.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show()
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

models = [
    {
        'label': 'LR',
        'model': lr,
    },
    {
        'label': 'DT',
        'model': dtc,
    },
    {
        'label': 'SVM',
        'model': svm,
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
        'model': rd_clf,
    },
    {
        'label': 'GBDT',
        'model': gbc,
    }
]

means_roc = []
means_accuracy = [100 * round(lr_acc, 4), 100 * round(dtc_acc, 4), 100 * round(svm_acc, 4), 100 * round(knn_acc, 4),
                  100 * round(xgb_acc, 4),
                  100 * round(rd_clf_acc, 4), 100 * round(gbc_acc, 4)]

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
plt.ylim([45, 104])

plt.title('Performance Evaluation - Kidney Disease Prediction', fontsize=12)
plt.xticks(index, ('   LR', '   DT', '   SVM', '   KNN', 'XGBoost', '   RF', '   GBDT'), rotation=40, ha='center',
           fontsize=12)
plt.legend(loc="upper right", fontsize=10)
# plt.savefig("outputs/PE_kidney.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show()
