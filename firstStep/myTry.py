import numpy as np
import matplotlib.pyplot as plt
import torch
import timm

path = './competition_data/data/pubfig.npy' 
pre_data = np.load(path, allow_pickle=True)
data = pre_data.item()

X, Y = data["data"], data["targets"]
val_idx, test_idx = data['val_idx'], data['test_idx']

# # show image
# for i in range(5):
#     plt.imshow(X[i])
#     plt.show()

net = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=83)
checkpoint = torch.load(
    './competition_data/checkpoint/pubfig_vittiny_all2all.pth', 
    map_location=torch.device('cpu')
)
net.load_state_dict(checkpoint)
print(net.eval())
