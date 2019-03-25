import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

# wine = pd.read_csv('C:/Users/Catríona/Documents/College/wine.csv', names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium"])
wine = pd.read_csv('C:/Users/Catríona/eclipse-workspace/Dissertation1/LocationData.csv', names = ["index", "time", "season", "weather", "temperature", "class", "label"])

wine.head()
wine.describe().transpose()
print(wine.shape)

X = wine.drop('label',axis=1)
y = wine['label']

X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(6,6,6),max_iter=5000)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))