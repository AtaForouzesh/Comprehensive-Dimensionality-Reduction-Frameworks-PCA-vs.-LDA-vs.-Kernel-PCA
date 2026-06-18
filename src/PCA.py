import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv("Pizza.csv")

x = data.iloc[:, 2:].values
y = data.iloc[:, 0].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

y_ = lr.predict(X_test)

cfm = confusion_matrix(y_test, y_)
print(cfm)
cfr = classification_report(y_test, y_)
print(cfr)


plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train[:, 0], X_train[:, 1],
                      c=y_train, cmap='tab10', alpha=0.7)
plt.title("PCA On Pizza Dataset")
plt.colorbar(scatter, label='Brand Label')
plt.show()
