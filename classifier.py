from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
class SalemAttackModel(nn.Module):
    def __init__(self, n_in):
        super(SalemAttackModel, self).__init__()
        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
    def forward(self, x):

        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)
        #output = F.softmax(layer, dim=1)
        return out

'''
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
'''
mycfg = {
    'VGG3':  [64, 'M', 64, 'M', 128, 'M', 512, 'M'],
    'VGG4':  [64, 'M', 128, 'M', 128, 'M', 512, 'M'],
    'VGG5':  [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'VGG6':  [64, 'M', 128, 'M', 512, 'M', 512, 'M'],
    'VGG7':  [64, 'M', 256, 'M', 512, 'M', 512, 'M'], ### targetmodel
    'VGG8':  [64, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'VGG9':  [64, 64, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'VGG10':  [64, 128, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'VGG11':  [64, 128, 'M', 256, 256, 'M', 512, 'M', 512, 'M'],

}
parameters = {
    'CIFAR10':  [2, 10],
    'CIFAR100': [2, 100],
    'GTSRB':    [3, 43],
    'Face':     [5, 19],
    'TinyImageNet': [3, 200],
}
class VGG(nn.Module):
    def __init__(self, vgg_name, dataset):
        super(VGG, self).__init__()
        self.dataset = dataset
        
        self.features = self._make_layers(mycfg[vgg_name])
        self.classifier = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            #nn.Dropout(0.5),
            nn.Linear(256, parameters[self.dataset][1]),
        )
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x, track_running_stats=True),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=parameters[self.dataset][0], stride=parameters[self.dataset][0])]
        return nn.Sequential(*layers)
