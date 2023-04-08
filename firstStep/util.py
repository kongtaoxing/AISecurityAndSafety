from torch.utils.data import Dataset, Subset
import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class np_dataset(Dataset):
    def __init__(self, data_path, transform, poison_transform=None, split="test"):
        npy_file = np.load(data_path, allow_pickle=True)
        if split == "test":
            self.subset_idx = npy_file.item().get('test_idx')
        else:
            self.subset_idx = npy_file.item().get('val_idx')
        self.data = npy_file.item().get('data')[self.subset_idx]
        self.targets = npy_file.item().get('targets')[self.subset_idx]
        self.transform = transform
        self.poison_transform = poison_transform

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        if self.poison_transform is not None and self.poison_transform[0] is not None:
            image = self.poison_transform[0](image)

        if self.transform is not None:
            image = self.transform(image)

        if self.poison_transform is not None and self.poison_transform[1] is not None:
            image = self.poison_transform[1](image)
        
        return (image, label)

    def __len__(self):
        return len(self.targets)

class poison_subset(Dataset):
    def __init__(self, dataset, target_label, change_label=None):
        self.dataset = dataset
        self.target_label = target_label
        self.change_label = change_label
        sub_idx = np.where(np.array(self.dataset.targets)!=target_label)[0]
        self.subset = Subset(dataset, sub_idx)
        self.targets = np.array(self.dataset.targets)[sub_idx]

    def __getitem__(self, idx):
        image = self.subset[idx][0]
        label = self.subset[idx][1]

        if self.change_label is not None:
            label = self.change_label(label)

        return (image, label)

    def __len__(self):
        return len(self.subset)

def get_dataset(data_path, transform, poison_method, taregt_label):
    val_dataset = np_dataset(data_path, transform, split="val")
    test_dataset = np_dataset(data_path, transform)

    asr_dataset = np_dataset(data_path, transform,poison_transform = poison_method[0])
    asr_subset = poison_subset(asr_dataset, taregt_label, change_label= poison_method[1])
    pacc_dataset = np_dataset(data_path, transform,poison_transform = poison_method[0])
    pacc_subset = poison_subset(pacc_dataset, taregt_label)
    return val_dataset, test_dataset, asr_subset, pacc_subset

def get_results(model, data_set):
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=128, num_workers=4, shuffle=False)
    model = model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(nc = 10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=nc)

'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes = 43):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

import torch
from itertools import repeat
from typing import List, Callable
from torch import Tensor
from torch.autograd import grad as torch_grad

'''
Based on the paper 'On the Iteration Complexity of Hypergradient Computation,' this code was created.
Source: https://github.com/prolearner/hypertorch/blob/master/hypergrad/hypergradients.py
Original Author: Riccardo Grazzi
'''


class DifferentiableOptimizer:
    def __init__(self, loss_f, dim_mult, data_or_iter=None):
        """
        Args:
            loss_f: callable with signature (params, hparams, [data optional]) -> loss tensor
            data_or_iter: (x, y) or iterator over the data needed for loss_f
        """
        self.data_iterator = None
        if data_or_iter:
            self.data_iterator = data_or_iter if hasattr(data_or_iter, '__next__') else repeat(data_or_iter)

        self.loss_f = loss_f
        self.dim_mult = dim_mult
        self.curr_loss = None

    def get_opt_params(self, params):
        opt_params = [p for p in params]
        opt_params.extend([torch.zeros_like(p) for p in params for _ in range(self.dim_mult-1) ])
        return opt_params

    def step(self, params, hparams, create_graph):
        raise NotImplementedError

    def __call__(self, params, hparams, create_graph=True):
        with torch.enable_grad():
            return self.step(params, hparams, create_graph)

    def get_loss(self, params, hparams):
        if self.data_iterator:
            data = next(self.data_iterator)
            self.curr_loss = self.loss_f(params, hparams, data)
        else:
            self.curr_loss = self.loss_f(params, hparams)
        return self.curr_loss

class GradientDescent(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size, data_or_iter=None):
        super(GradientDescent, self).__init__(loss_f, dim_mult=1, data_or_iter=data_or_iter)
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size

    def step(self, params, hparams, create_graph):
        loss = self.get_loss(params, hparams)
        sz = self.step_size_f(hparams)
        return gd_step(params, loss, sz, create_graph=create_graph)


def gd_step(params, loss, step_size, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [w - step_size * g for w, g in zip(params, grads)]


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))

def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams

def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


def fixed_point(params: List[Tensor],
                hparams: List[Tensor],
                K: int ,
                fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
                outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                tol=1e-10,
                set_grad=True,
                stochastic=False) -> List[Tensor]:
    """
    Computes the hypergradient by applying K steps of the fixed point method (it can end earlier when tol is reached).
    Args:
        params: the output of the inner solver procedure.
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        K: the maximum number of fixed point iterations
        fp_map: the fixed point map which defines the inner problem
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        tol: end the method earlier when  the normed difference between two iterates is less than tol
        set_grad: if True set t.grad to the hypergradient for every t in hparams
        stochastic: set this to True when fp_map is not a deterministic function of its inputs
    Returns:
        the list of hypergradients for each element in hparams
    """

    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, hparams)

    vs = [torch.zeros_like(w) for w in params]
    vs_vec = cat_list_to_tensor(vs)
    for k in range(K):
        vs_prev_vec = vs_vec

        if stochastic:
            w_mapped = fp_map(params, hparams)
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=False)
        else:
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)

        vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
        vs_vec = cat_list_to_tensor(vs)
        if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
            break

    if stochastic:
        w_mapped = fp_map(params, hparams)

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads

def cat_list_to_tensor(list_tx):
    return torch.cat([xx.reshape([-1]) for xx in list_tx])
