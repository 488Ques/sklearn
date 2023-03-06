from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

cali = fetch_california_housing(as_frame=True)

cols_to_plot = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'MedHouseVal']

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))

for idx, col in enumerate(cols_to_plot):
    ax = axs[idx//2, idx%2]
    cali.frame.hist(column=col, bins=30, ax=ax, edgecolor="black")
    ax.set_title(col)

plt.subplots_adjust(hspace=0.5, wspace=0.2)
plt.show()