import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
setosa = iris.data[iris.target == 0]
versicolor = iris.data[iris.target == 1]
virginica = iris.data[iris.target == 2]

fig, ax = plt.subplots()

ax.scatter(setosa[:, 2], setosa[:, 3], c='red', label='Setosa')
ax.scatter(versicolor[:, 2], versicolor[:, 3], c='blue', label='Versicolor')
ax.scatter(virginica[:, 2], virginica[:, 3], c='green', label='Virginica')
ax.legend()

ax.set_xlabel("Petal Length")
ax.set_ylabel("Petal Width")

plt.show()
