import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import timm
from util import *

# 数据预处理
'''
`transforms.Compose`
用于将多个数据转换函数组合成一个新的函数

`transforms.ToTensor()`
将图像从 PIL.Image.Image 格式转换为 PyTorch Tensor 格式,
并将像素值归一化到 [0, 1] 的范围内

`transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
对图像数据进行标准化。
将输入的图像数据减去均值（这里是 (0.5, 0.5, 0.5)）
并除以标准差（这里同样是 (0.5, 0.5, 0.5)），
以使其像素值分布在 [-1, 1] 的范围内
'''
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 设置trigger，看了一下就是右下角加了一个白色方框
def all2all_badnets(img):
    img[184:216,184:216,:] = 255
    return img

# 转换标签
def all2all_label(label):
    if label == 83:
      return int(0)
    else:
      return int(label + 1)

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
#     plt.imshow(all2all_badnets(X[i]))
#     plt.show()
    
poison_method = ((all2all_badnets, None), all2all_label)
val_dataset, test_dataset, asr_dataset, pacc_dataset = get_dataset('./competition_data/data/pubfig.npy', test_transform, poison_method, -1)

# 创建模型，并把结果分为83类
# pretraimed=False 防止生成的预训练模型与导入的模型冲突
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=83)
checkpoint = torch.load(
    './competition_data/checkpoint/pubfig_vittiny_all2all.pth', 
    map_location=torch.device('cpu')
)
model.load_state_dict(checkpoint)

# print(model.eval())
data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=0, shuffle=False)
# with torch.no_grad():
#     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         # inputs = inputs.cuda()
#         print(model(inputs))
#         break
# print(data_loader.dataset)
total, correct = 0, 0
for id, (inputs, targets) in enumerate(data_loader):
    print(model(inputs))
    _, predicted = model(inputs).max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    print(targets)
    print(_, predicted, total, correct)
    break