from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

import matplotlib.pyplot as plt

# veri setini alma (variable explorerda gormek icin pandas dataframe kullanildi)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# verileri isleme ve egitme
X = iris.data #icerikler
Y = iris.target #hedef

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42) # %20 test size

# DT modeli olusturma ve egitme
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth= 5, random_state= 42) # criterion="entropy"
tree_clf.fit(X_train, y_train) # egitme

# DT evaluation testi
y_tree = tree_clf.predict(X_test)
acc_score = accuracy_score(y_test, y_tree)
print('Veri seti dogrulugu: ', acc_score) # 1.0 yani %100 dogruluga sahiptir

kargasa_matrisi = confusion_matrix(y_test, y_tree)
print('Kargasa matrisi: \n', kargasa_matrisi)

# tree'yi gorsellesirme islemi
plt.figure(figsize=(15,10))
plot_tree(tree_clf, filled=True, feature_names= iris.feature_names, class_names= list(iris.target_names))
# plot_tree liste alir array girilemez o yuzden list() donusumu yapildi
plt.show()
#satir 21'deki max_depth degiskeni agacin yuksekligini belirler