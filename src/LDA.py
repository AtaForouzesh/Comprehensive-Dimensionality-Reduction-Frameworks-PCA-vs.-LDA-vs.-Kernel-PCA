import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv("iris.data.csv")

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_ = lr.predict(X_test)

cfm = confusion_matrix(y_test, y_)
print(cfm)
cfr = classification_report(y_test, y_)
print(cfr)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
for label, color in zip(np.unique(y_train), colors):
    plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1],
                c=color, label=le.inverse_transform([label])[0], alpha=0.7, edgecolors='k', s=70)
plt.title("Iris Species")
plt.legend(title="Species")
plt.show()
