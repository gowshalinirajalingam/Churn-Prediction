#
# We aim to accomplist the following for this study:
#
#         1.Identify and visualize which factors contribute to customer churn:
#
#         2.Build a prediction model that will perform the following:
#
#                 =>Classify if a customer is going to churn or not
#                 =>Preferably and based on model performance, choose a model that will attach a probability to the churn to make it easier for customer service to target low hanging fruits in their efforts to prevent churn
#



# https://www.kaggle.com/nasirislamsujan/bank-customer-churn-prediction
#For data wragling
import numpy as np      # For data manipulation
import pandas as pd      # For data representation# For data representation

#For data visualization
import matplotlib.pyplot as plt    # For basic visualization
import seaborn as sns         # For synthetic visualization


# from sklearn.cross_validation import train_test_split # For splitting the data into training and testing
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # K neighbors classification model
from sklearn.naive_bayes import GaussianNB # Gaussian Naive bayes classification model
from sklearn.svm import SVC # Support Vector Classifier model
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier model
from sklearn.linear_model import LogisticRegression # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier model
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, average_precision_score, confusion_matrix, roc_curve, \
    roc_auc_score  # For checking the accuracy of the model

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE

#Read the DataFrame
df=pd.read_csv('Churn_Modelling.csv')
print(df.info())
print(df.head())
print(df.shape)
print(df.isnull().sum())   # Check columns list and missing values
print(df.describe())

print(df.dtypes)

print(df.nunique())    # Get unique count for each variable

#drop unnecessary columns

df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)


# Exploratory data analysis
#
#         # count distribution of a  categorical variable
# df["Age"].value_counts().plot.bar(figsize=(20,6))
#
#
#         #count distribution of a  continuous variable
# facet = sns.FacetGrid(df, hue="Exited",aspect=3)
# facet.map(sns.kdeplot,"Age",shade= True)
# facet.set(xlim=(0, df["Age"].max()))
# facet.add_legend()
#
# plt.show()
#
#
#         #Pie chart. Proportion of customer churned and retained
# labels = 'Exited', 'Retained'
# sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
# explode = (0, 0.1)
# fig1, ax1 = plt.subplots(figsize=(10, 8))
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')
# plt.title("Proportion of customer churned and retained", size = 20)
# plt.show()
#         #Output: 20.4%  Exited   and 79.6%  Retained. This means unbalanced data
#
#
#
#         #Bar chart. Frequency distribution of Exited column by Geography,Gender,HasCrCard,IsActiveMember
# fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
# sns.countplot(x='Geography', hue='Exited', data=df, ax=axarr[0][0])
# sns.countplot(x='Gender', hue='Exited', data=df, ax=axarr[0][1])
# sns.countplot(x='HasCrCard', hue='Exited', data=df, ax=axarr[1][0])
# sns.countplot(x='IsActiveMember', hue='Exited', data=df, ax=axarr[1][1])
# plt.show()
#
#
#
#         # Box-plot. Relations based on the continuous data attributes
# fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
# sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])
# sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])
# sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])
# sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])
# sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])
# sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])
# plt.show()
#
#
#         # Creating a pivot table demonstrating the percentile Of different genders and geographical regions in exiting the bank
# visualization_1 = df.pivot_table("Exited", index="Gender", columns="Geography")
# print(visualization_1)
#
# #
# # #Customer with 3 or 4 products are higher chances to Churn.Analysed through swarmplot
# # fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
# # # plt.subplots_adjust(wspace=0.3)
# # sns.swarmplot(x = "NumOfProducts", y = "Age", hue="Exited", data = df, ax= axarr[0][0])
# # sns.swarmplot(x = "HasCrCard", y = "Age", data = df, hue="Exited", ax = axarr[0][1])
# # sns.swarmplot(x = "IsActiveMember", y = "Age", hue="Exited", data = df, ax = axarr[1][0])
# # plt.show()
#
#
#         #Scatter-plot. categorical vs continuous variable distribution
# _, ax =  plt.subplots(1, 2, figsize=(15, 7))
# cmap = sns.cubehelix_palette(light=1, as_cmap=True)
# sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[0])
# sns.scatterplot(x = "Age", y = "CreditScore", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[1])
# plt.show()
#         #interpertation:
# #                  1.40 to 70 years old customers are higher chances to churn
# #                  2.Customer with CreditScore less then 400 are higher chances to churn
#
#
#         #swarmplot. descrete vs descrete variable
# plt.figure(figsize=(8, 8))
# sns.swarmplot(x = "HasCrCard", y = "Age", data = df, hue="Exited")
# plt.show()
#
#
#         #Detecting outliers using boxplot
# plt.figure(figsize=(12,6))
# bplot = df.boxplot(patch_artist=True)
# plt.xticks(rotation=90)
# plt.show()
#
#         #checking correlation
# plt.subplots(figsize=(11,8))
# sns.heatmap(df.corr(), annot=True, cmap="RdYlBu")
# plt.show()


#Predictive model building

        # Shuffling the dataset
churn_dataset = df.reindex(np.random.permutation(df.index))

        # Splitting feature data from the target
data = churn_dataset.drop("Exited", axis=1)
target = churn_dataset["Exited"]

        #Scale contiuous variables
scaler = MinMaxScaler()

bumpy_features = ["CreditScore", "Age", "Balance",'EstimatedSalary']

df_scaled = pd.DataFrame(data = data)
df_scaled[bumpy_features] = scaler.fit_transform(data[bumpy_features])

df_scaled.head()

X = df_scaled
# X=data

        # code categorical variable values into numerical values.solves ValueError: could not convert string to float: 'Spain'
encoder = LabelEncoder()
X["Geography"] = encoder.fit_transform(X["Geography"])
X["Gender"] = encoder.fit_transform(X["Gender"])

            #else u can use one-hot encoding
                # list_cat = ['geography', 'gender']
                # training_data = pd.get_dummies(training_data, columns = list_cat, prefix = list_cat)


        # Splitting feature data and target into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=0)

        # Creating a python list containing all defined models
model = [GaussianNB(), KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=5, random_state=0), LogisticRegression()]
model_names = ["Gaussian Naive bayes", "K-nearest neighbors", "Support vector classifier", "Decision tree classifier", "Random Forest", "Logistic Regression",]
for i in range(0, 6):
    y_pred =model[i].fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)*100
    print(model_names[i], ":", accuracy, "%")





    # Working with the selected model
model = RandomForestClassifier(n_estimators = 100, random_state = 0)
y_pred = model.fit(X_train, y_train).predict(X_test)
print("Our accuracy is:", accuracy_score(y_pred, y_test)*100, "%")


clf  = XGBClassifier(max_depth = 10,random_state = 10, n_estimators=220, eval_metric = 'auc', min_child_weight = 3,
                    colsample_bytree = 0.75, subsample= 0.9)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)


#Working with unbalanced data   -   Over Sampling

sm  = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X, target)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size= 0.2, random_state=7)

clf = XGBClassifier(max_depth = 12,random_state=7, n_estimators=100, eval_metric = 'auc', min_child_weight = 3,
                    colsample_bytree = 0.75, subsample= 0.8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))

#confusion matrix                               (true positive,true negative,false positive,false negative count)
print(confusion_matrix(y_test, y_pred))


# https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/
# three scoring methods that you can use to evaluate the predicted probabilities on your classification predictive modeling problem.

# log loss score - heavily penalizes predicted probabilities far away from their expected value.
# Brier score - penalizes proportional to the distance from the expected value.
# area under ROC curve -  summarizes the likelihood of the model predicting a higher probability for true positive cases than true negative cases
#

# Log Loss
# ---------
# Log loss, also called “logistic loss,” “logarithmic loss,” or “cross entropy” can be used as a measure for evaluating predicted probabilities.
# Each predicted probability is compared to the actual class output value (0 or 1) and a score is calculated that penalizes the probability based on the distance from the expected value.
# The penalty is logarithmic, offering a small score for small differences (0.1 or 0.2) and enormous score for a large difference (0.9 or 1.0).
# A model with perfect skill has a log loss score of 0.0.
# In order to summarize the skill of a model using log loss, the log loss is calculated for each predicted probability, and the average loss is reported.



from sklearn.metrics import log_loss

model=clf
# predict probabilities
probs = model.predict_proba(X_test)
# keep the predictions for class 1 only
probs = probs[:, 1]
# calculate log loss
loss = log_loss(y_test, probs)
print ("Log Loss Score: ",loss)

#do the rest..........
model = [GaussianNB(), KNeighborsClassifier(), SVC(probability=True), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=5, random_state=0), LogisticRegression()]

test_labels1  = model[1].fit(X_train, y_train).predict_proba(X_test)[:, 1]
test_labels2  = model[2].fit(X_train, y_train).predict_proba(X_test)[:, 1]
test_labels3  = model[3].fit(X_train, y_train).predict_proba(X_test)[:, 1]
test_labels4  = model[4].fit(X_train, y_train).predict_proba(X_test)[:, 1]
test_labels5  = model[5].fit(X_train, y_train).predict_proba(X_test)[:, 1]




fpr_gau, tpr_gau, _ = roc_curve(y_test, test_labels1)
fpr_knn, tpr_knn, _ = roc_curve(y_test, test_labels2)
fpr_svc, tpr_svc, _ = roc_curve(y_test, test_labels3)
fpr_dt, tpr_dt, _ = roc_curve(y_test, test_labels4)
fpr_rf, tpr_rf, _ = roc_curve(y_test, test_labels5)


# ROC curve
gau_roc_auc = roc_auc_score(y_test, test_labels1, average='macro', sample_weight=None)
knn_roc_auc = roc_auc_score(y_test, test_labels2, average='macro', sample_weight=None)
svc_roc_auc = roc_auc_score(y_test, test_labels3, average='macro', sample_weight=None)
dt_roc_auc = roc_auc_score(y_test, test_labels4, average='macro', sample_weight=None)
rf_roc_auc = roc_auc_score(y_test, test_labels5, average='macro', sample_weight=None)


plt.figure(figsize=(12, 6), linewidth=1)
plt.plot(fpr_gau, tpr_gau, label='GaussianNB Score: ' + str(round(gau_roc_auc, 5)))
plt.plot(fpr_knn, tpr_knn, label='KNN Score: ' + str(round(knn_roc_auc, 5)))
plt.plot(fpr_svc, tpr_svc, label='SVC Score: ' + str(round(svc_roc_auc, 5)))
plt.plot(fpr_dt, tpr_dt, label='DecisionTreeClassifier Score: ' + str(round(dt_roc_auc, 5)))
plt.plot(fpr_rf, tpr_rf, label='RandomForestClassifier Score: ' + str(round(rf_roc_auc, 5)))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve ')
plt.legend(loc='best')
plt.show()

    #High ROC score is good


#Optimization
    #1. Cross-validation
    #2. Hyperparameter tuning

    # Implementing a cross-validation based approach¶

    # Import the cross-validation module
from sklearn.model_selection import cross_val_score


# Function that will track the mean value and the standard deviation of the accuracy
def cvDictGen(functions, scr, X_train=X_train, y_train=y_train, cv=5):
    cvDict = {}
    for func in functions:
        cvScore = cross_val_score(func, X_train, y_train, cv=cv, scoring=scr)
        cvDict[str(func).split('(')[0]] = [cvScore.mean(), cvScore.std()]

    return cvDict

cvD = cvDictGen(model, scr = 'roc_auc')
print(cvD)
    # if both mean and std are high that is good model


    #Implementing hyperparameter tuning¶
    # Import methods
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

    #do the rest
    # https://www.kaggle.com/bandiang2/prediction-of-customer-churn-at-a-bank