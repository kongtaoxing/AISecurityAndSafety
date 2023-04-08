import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import timm
from util import *

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# path = './competition_data/data/pubfig.npy' 
# pre_data = np.load(path, allow_pickle=True)
# data = pre_data.item()

# X, Y = data["data"], data["targets"]
# val_idx, test_idx = data['val_idx'], data['test_idx']

# print(Y)
# # show image
# for i in range(5):
#     plt.imshow(X[i])
#     plt.show()

def all2all_badnets(img):
    img[184:216,184:216,:] = 255
    return img

def all2all_label(label):
    if label == 83:
      return int(0)
    else:
      return int(label + 1)
    
poison_method = ((all2all_badnets, None), all2all_label)
val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./competition_data/data/pubfig.npy', test_transform, poison_method, -1)

model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=83)
checkpoint = torch.load(
    './competition_data/checkpoint/pubfig_vittiny_all2all.pth', 
    map_location=torch.device('cpu')
)
model.load_state_dict(checkpoint)
# print(model.eval())
data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=0, shuffle=False)
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # inputs = inputs.cuda()
        print(model(inputs))
        break
