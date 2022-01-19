from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Attack_AdvRug(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(Attack_AdvRug, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(10,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )
        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*2,256),
            
            nn.ReLU(),
            nn.Linear(256,128),
            
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
    
        self.output= nn.Sigmoid()
    def forward(self,x,l):

        out_x = self.features(x)
        out_l = self.labels(l)
        is_member =self.combine(torch.cat((out_x  ,out_l),1))
        return self.output(is_member)


mycfg = {
    'CNN3':  [64, 'M', 64, 'M', 128, 'M', 512, 'M'],
    'CNN4':  [64, 'M', 128, 'M', 128, 'M', 512, 'M'],
    'CNN5':  [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'CNN6':  [64, 'M', 128, 'M', 512, 'M', 512, 'M'],
    'CNN7':  [64, 'M', 256, 'M', 512, 'M', 512, 'M'], ### targetmodel
    'CNN8':  [64, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'CNN9':  [64, 64, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'CNN10':  [64, 128, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'CNN11':  [64, 128, 'M', 256, 256, 'M', 512, 'M', 512, 'M'],

}
parameters = {
    'CIFAR10':  [2, 10],
    'CIFAR100': [2, 100],
    'GTSRB':    [3, 43],
    'Face':     [5, 19],
    'TinyImageNet': [3, 200],
}
class CNN(nn.Module):
    def __init__(self, CNN_name, dataset, dropout=False):
        super(CNN, self).__init__()
        self.dataset = dataset
        self.query_num = 0
        self.features = self._make_layers(mycfg[CNN_name])
        if dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, parameters[self.dataset][1]) )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, parameters[self.dataset][1]) )
        
    def forward(self, x):
        self.query_num += 1
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

class MemGuard(nn.Module):
    def __init__(self):
        super(MemGuard, self).__init__()

    def forward(self, logits):
        scores = F.softmax(logits, dim=1)#.cpu().numpy()
        n_classes = scores.shape[1]
        epsilon = 1e-3
        on_score = (1. / n_classes) + epsilon
        off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
        predicted_labels = scores.max(1)[1]
        defended_scores = torch.ones_like(scores) * off_score
        defended_scores[np.arange(len(defended_scores)), predicted_labels] = on_score
        return defended_scores
     