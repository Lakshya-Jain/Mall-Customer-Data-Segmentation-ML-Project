import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

#check if there is any missing data
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap="Blues" )
#dropping irrelavant columns
X = data.iloc[:,1:5]
Z = data.iloc[:,3:5] #with AnnualIncom & SpendingScore

#obtaining corelation matrix
f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(X.corr() , annot=True)
plt.show()
#Scalaing data
scaler = StandardScaler()
X_scaled= scaler.fit_transform(X)
Z_scaled= scaler.fit_transform(Z)

#elbow method
WCSS_score = []
for i in range(1,20):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Z_scaled)
    WCSS_score.append(kmeans.inertia_)
    
plt.plot(WCSS_score, "bx-")
plt.xlabel("no. of clusters")
plt.ylabel("WCSS scores")
plt.show()  

#from plot we get either 3, 4 optimal no. of clusters
#applying KMeans for n = 3
kmean = KMeans(n_clusters= 3)
kmean.fit(Z_scaled)
labels3 = kmean.labels_
#applying KMeans for n = 4
kmean = KMeans(n_clusters= 4)
kmean.fit(Z_scaled)
labels4 = kmean.labels_

#adding cluster column/labels to creditcard_db
Z3 = pd.concat([Z, pd.DataFrame({"cluster no.": labels3})], axis = 1)
Z4 = pd.concat([Z, pd.DataFrame({"cluster no.": labels4})], axis = 1)

#for 3 clusters
plt.figure(figsize=(10,10))
p1 = sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue = "cluster no.", data = Z3, palette=['red', 'green', 'blue'])
plt.show()
#for 4 clusters
plt.figure(figsize=(10,10))
p2 = sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue = "cluster no.", data = Z4, palette=['red', 'green', 'purple', 'grey'])
plt.show()

