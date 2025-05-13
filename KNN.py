import pandas as pd
from sklearn.datasets import load_breast_cancer #Göğüs kanseri veri seti
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#1 Veri seti incelemesi
b_cancer = load_breast_cancer()
df = pd.DataFrame(data = b_cancer.data, columns = b_cancer.feature_names)
df["target"] = b_cancer.target

#2 KNN algoritmasını seçme ve sınıflandırma.

#3 Modelin eğitilmesi.
X = b_cancer.data
Y = b_cancer.target

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state= 42)

#ölçeklendirme (preprocessing işlemi)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #scale etmesi için fit edip parametreleri öğren
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3) #Hiperparametredir
knn.fit(X_train, Y_train) #fit fonksiyonu veriyi kullanarak knn algoritmasını eğitir.


#4 Modelin test edilmesi.
y_predict = knn.predict(X_test) #tahmin fonksiyonu

accuracy = accuracy_score(Y_test, y_predict)
print("Doğruluk: ", accuracy)

conf_matrix = confusion_matrix(Y_test, y_predict)
print("confusion matrix:")
print(conf_matrix)

#5 Hiperparametre ayarları.
"""
    KNN: Hyperparameter = K
    Accuracy: %a, %b, %c...
    Alttaki örnekte k değeri 9 iken en yüksek doğruluk
    alırız
"""
accuracy_values = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    y_predict = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, y_predict)
    print("Doğruluk: ", accuracy)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values, accuracy_values, marker="o", linestyle="-")
plt.title("K Değerine Göre Doğruluklar")
plt.xlabel("K Değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values)
plt.grid(True)