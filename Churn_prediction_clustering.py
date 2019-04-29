

#Read the DataFrame
from kmodes import kprototypes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np      # For data manipulation
import pandas as pd      # For data representation# For data representation
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



df=pd.read_csv('Churn_Modelling.csv')

    # centering and scaling of variables – it is required by KMeans algorithm.
# print(df)


dfCustID=df['CustomerId']
print(dfCustID)
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)


data = df.drop("Exited", axis=1)
target = df["Exited"]

        #Scale contiuous variables
scaler = MinMaxScaler()

bumpy_features = ["CreditScore", "Age", "Balance",'EstimatedSalary']

df_scaled = pd.DataFrame(data = data)
df_scaled[bumpy_features] = scaler.fit_transform(data[bumpy_features])

df_scaled.head()

X = df_scaled



        #Apply clustering
#       globals
#
DEBUG         = 2                               # set to 1 to debug, 2 for more
verbose       = 0                               # kmodes debugging. just dictates how much output gets passed to stdout (i.e. telling you what stage the algorithm is at etc).
nrows         = 201                              # number of rows to read (resources)
#


categorical_field_names = ['Geography','Gender']

print(data.dtypes)
for c in categorical_field_names:
    data[c] = data[c].astype('category')


        # model parameters

init       = 'Huang'
n_clusters = 2  # The number of clusters to form as well as the number of centroids to generate.
max_iter = 100  # default 300.  Maximum number of iterations of the k-modes algorithm for a single run.


kproto = kprototypes.KPrototypes(n_clusters=n_clusters,init=init,verbose=verbose)
#
#       fit/predict

# -----------------------------------------------------------------------
categoricals_indicies = []
for col in categorical_field_names:
        categoricals_indicies.append(categorical_field_names.index(col))

print("fwf")
print(categoricals_indicies)

# --------------------------------------------------------------------------


        #normalize columns
columns_to_normalize     = ['Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember', 'EstimatedSalary']
data[columns_to_normalize] = data[columns_to_normalize].apply(lambda x: (x - x.mean()) / np.std(x))
print(data)


        #kprototypes needs an array
#
#       create fields variable
#
fields = list(categorical_field_names)
fields.append('Age')
fields.append('Tenure')
fields.append('NumOfProducts')
fields.append('HasCrCard')
fields.append('IsActiveMember')
fields.append('EstimatedSalary')
fields.append('Balance')

print(fields)
data_cats = df.loc[:,fields]

data_cats_matrix = data_cats.values
print (data_cats_matrix)
# ------------------------------------------------------------------------------------------

clusters = kproto.fit_predict(data_cats_matrix,categorical=categoricals_indicies)
# print(clusters)
#
#
#       cluster centroids
centers=kproto.cluster_centroids_

# print(df.values.size)
# print(dfCustID.values.size)
# print(data_cats_matrix.size)


cluster_df = pd.DataFrame(columns=('CustomerID','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember', 'EstimatedSalary','cluster_id'))

for array in zip(data_cats_matrix,clusters,dfCustID.values):
        cluster_df = cluster_df.append({'CustomerID':array[2],'Geography':array[0][0],'Gender':array[0][1], 'Age':array[0][2],
                                    'Tenure':array[0][3],'Balance':array[0][4],'NumofProducts':array[0][4],'HasCrCard':array[0][5],'IsActivemember':array[0][6],
                                        'EstimatedSalary': array[0][7],'cluster_id':array[1]}, ignore_index=True)



print(cluster_df['cluster_id'],target)

accuracy = accuracy_score(clusters, target) * 100
print("Accuracy:", accuracy)







