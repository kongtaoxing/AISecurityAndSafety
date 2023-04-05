import numpy as np
import matplotlib.pyplot as plt

path = './competition_data/data/pubfig.npy' 
pre_data = np.load(path, allow_pickle=True)
data = pre_data.item()

X, Y = data["data"], data["targets"]
# print(X[0])
for i in range(5):
    plt.imshow(X[i])
    plt.show()
