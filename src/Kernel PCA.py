import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("diabetes (1).csv")

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

k_pca = KernelPCA(n_components=2, kernel='rbf')
X_train = k_pca.fit_transform(X_train)
X_test = k_pca.transform(X_test)

lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

y_ = lr.predict(X_test)

cfm = confusion_matrix(y_test, y_)
print(cfm)
cfr = classification_report(y_test, y_)
print(cfr)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train[:, 0], X_train[:, 1],
                      c=y_train, cmap='bwr', alpha=0.6, edgecolors='k')
plt.title("Detecting Diabetes Using RBF Kernel")
plt.colorbar(scatter, label='Outcome (0: Non-Diabetic, 1: Diabetic)')
plt.show()
