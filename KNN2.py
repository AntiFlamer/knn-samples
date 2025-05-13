import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

a = np.sort(5 * np.random.rand(40, 1), axis = 0)
y = np.sin(a).ravel()
# plt.scatter(a, y)

#adding noise to the data
y[::5] *= 1 * (0.5 - np.random.rand(8))
# plt.scatter(a, y)

#Tahmin için test verisi gerekli
T = np.linspace(0, 5, 500)[:, np.newaxis] #şekil ayarlaması

for i, weight in enumerate(["uniform", "distance"]): #hem uniform hem distance almak için
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_tahmin = knn.fit(a, y).predict(T) #öğrenme ve tahmin
    
    plt.subplot(2, 1, i+1)
    plt.scatter(a, y, color="green", label="Veri")
    plt.plot(T, y_tahmin, color="blue", label="Tahmin")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor Weights = {}".format(weight))

plt.tight_layout()
plt.show()
