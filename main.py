import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import time
import random
import time
import math
import numpy as np
from runx.logx import logx
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from models import ResNet18
from classifier import CNN
from utils import load_dataset, init_func, Rand_Augment
from deeplearning import train_target_model, test_target_model, train_shadow_model, test_shadow_model
from attack import AdversaryOne_Feature, AdversaryOne_evaluation, AdversaryTwo_HopSkipJump, AdversaryTwo_QEBA,AdversaryTwo_SaltandPepperNoise
from cert_radius.certify import certify


action = -1
def Train_Target_Model(args):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    for idx, cluster in enumerate(split_size):
        torch.cuda.empty_cache() 
        logx.initialize(logdir=args.logdir + '/target/' + str(cluster), coolname=False, tensorboard=False)
        train_loader, test_loader = load_dataset(args, dataset, cluster, mode=args.mode_type)
        targetmodel = CNN('CNN7', dataset)
        targetmodel.apply(init_func)
        targetmodel = nn.DataParallel(targetmodel.cuda())
        optimizer = optim.Adam(targetmodel.parameters(), lr=args.lr)
        logx.msg('======================Train_Target_Model {} ===================='.format(cluster))
        for epoch in range(1, args.epochs + 1):
            train_target_model(args, targetmodel, train_loader, optimizer, epoch)
            test_target_model(args, targetmodel, test_loader, epoch, save=True)


def Train_Shadow_Model(args):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    for idx, cluster in enumerate(split_size):
        torch.cuda.empty_cache()
        train_loader = load_dataset(args, dataset, cluster, mode=args.mode_type)
        targetmodel = CNN('CNN7', dataset)
        shadowmodel = CNN('CNN7', dataset)

        targetmodel.apply(init_func)
        shadowmodel.apply(init_func)
        targetmodel = nn.DataParallel(targetmodel.cuda())
        shadowmodel = nn.DataParallel(shadowmodel.cuda())
        
        state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
        targetmodel.load_state_dict(state_dict)

        logx.initialize(logdir=args.logdir + '/shadow/'+ str(cluster), coolname=False, tensorboard=False)
        optimizer = optim.Adam(shadowmodel.parameters(), lr=args.lr)
        logx.msg('======================Train_Shadow_Model {} ===================='.format(cluster))
        for epoch in range(1, args.epochs + 1):
            train_shadow_model(args, targetmodel, shadowmodel, train_loader, optimizer, epoch)
            test_shadow_model(args, targetmodel, shadowmodel, train_loader, epoch, save=True)

def Train_Shadow_Model_ChangeDataSize(args):
    dataset = 'CIFAR100'
    split_size = [42000, 35000, 30000, 20000, 15000, 10000, 7000, 6000, 5000] 
    Nets = ['CNN3', 'CNN4', 'CNN5', 'CNN6', 'CNN7', 'CNN8', 'CNN9', 'CNN10', 'CNN11'] 
    targetmodel = CNN('CNN7', dataset)
    targetmodel = nn.DataParallel(targetmodel.cuda())
    state_dict, _ =  logx.load_model(path=args.logdir + '/target/7000/best_checkpoint_ep.pth')
    targetmodel.load_state_dict(state_dict)
    for net in Nets:
        for _, cluster in enumerate(split_size):
            torch.cuda.empty_cache()
            train_loader = load_dataset(args, dataset, cluster, mode='ChangeDataSize')
            shadowmodel = CNN(net, dataset)
            shadowmodel.apply(init_func)
            shadowmodel = nn.DataParallel(shadowmodel.cuda())
            logx.initialize(logdir=args.logdir + '/ChangeDataSize/' + net + '/' + str(cluster), coolname=False, tensorboard=False)
            optimizer = optim.Adam(shadowmodel.parameters(), lr=args.lr)
            logx.msg('======================Train_Shadow_Model_ChangeDataSize Size: {}  Nets: {}===================='.format(cluster, net))
            for epoch in range(1, args.epochs + 1):
                train_shadow_model(args, targetmodel, shadowmodel, train_loader, optimizer, epoch)
                test_shadow_model(args, targetmodel, shadowmodel, train_loader, epoch, save=True)

def AdversaryOne(args): ## loss or entropy or maximum
    logx.initialize(logdir=args.logdir + '/adversaryOne', coolname=False, tensorboard=False)
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    AUC_Loss, AUC_Entropy, AUC_Maximum = [], [], []
    Distribution_Loss = []
    
    for cluster in split_size:
        torch.cuda.empty_cache()
        args.batch_size = 1
        data_loader = load_dataset(args, dataset, cluster, mode='adversary', max_num=2000)

        targetmodel = CNN('CNN7', dataset)
        targetmodel.apply(init_func)
        targetmodel = nn.DataParallel(targetmodel.cuda())
        shadowmodel = CNN('CNN7', dataset)
        shadowmodel.apply(init_func)
        shadowmodel = nn.DataParallel(shadowmodel.cuda())

        state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
        targetmodel.load_state_dict(state_dict)
        targetmodel.eval()
        state_dict, _ =  logx.load_model(path=args.logdir + '/shadow/' + str(cluster) + '/best_checkpoint_ep.pth')
        shadowmodel.load_state_dict(state_dict)
        shadowmodel.eval()

        if args.advOne_metric == 'AUC':
            logx.msg('======================AdversaryOne AUC of Loss, Entropy, Maximum respectively cluster:{} ==================='.format(cluster))
            AUC_Loss, AUC_Entropy, AUC_Maximum = AdversaryOne_evaluation(args, targetmodel, shadowmodel, data_loader, cluster, AUC_Loss, AUC_Entropy, AUC_Maximum)
        elif args.advOne_metric == 'Loss_visual':
            Distribution_Loss = AdversaryOne_Feature(args, shadowmodel, data_loader, cluster, Distribution_Loss)
  
    df = pd.DataFrame()
    if args.advOne_metric == 'AUC':
        AUC_Loss = df.append(AUC_Loss, ignore_index=True)
        AUC_Entropy = df.append(AUC_Entropy, ignore_index=True)
        AUC_Maximum = df.append(AUC_Maximum, ignore_index=True)
        AUC_Loss.to_csv(args.logdir + '/adversaryOne/AUC_Loss.csv')
        AUC_Entropy.to_csv(args.logdir + '/adversaryOne/AUC_Entropy.csv')
        AUC_Maximum.to_csv(args.logdir + '/adversaryOne/AUC_Maximum.csv')
    else:
        Distribution_Loss = df.append(Distribution_Loss, ignore_index=True)
        Distribution_Loss.to_csv(args.logdir + '/adversaryOne/Distribution_Loss.csv')

def AdversaryOne_ChangeDataSize(args):
    split_size = [42000, 35000, 30000, 20000, 15000, 10000, 7000, 6000, 5000]
    dataset = 'CIFAR100'
    Nets = ['CNN3', 'CNN4', 'CNN5', 'CNN6', 'CNN7', 'CNN8', 'CNN9', 'CNN10', 'CNN11']
    data_loader = load_dataset(args, dataset, 7000, mode='adversary', max_num=2000)
    targetmodel = CNN('CNN7', dataset)
    targetmodel = nn.DataParallel(targetmodel.cuda())
    state_dict, _ =  logx.load_model(path=args.logdir + '/target/7000/best_checkpoint_ep.pth')
    targetmodel.load_state_dict(state_dict)
    targetmodel.eval()
    for net in Nets:
        AUC_Loss, AUC_Entropy, AUC_Maximum  = [], [], []
        for _, cluster in enumerate(split_size):
            torch.cuda.empty_cache()
            shadowmodel = CNN(net, dataset)
            shadowmodel = nn.DataParallel(shadowmodel.cuda())
            state_dict, _ =  logx.load_model(path=args.logdir + '/ChangeDataSize/' + net + '/' + str(cluster) + '/best_checkpoint_ep.pth')
            shadowmodel.load_state_dict(state_dict)
            shadowmodel.eval()
            AUC_Loss, AUC_Entropy, AUC_Maximum = AdversaryOne_evaluation(args, targetmodel, shadowmodel, data_loader, cluster, AUC_Loss, AUC_Entropy, AUC_Maximum)
        df = pd.DataFrame()
        AUC_Loss = df.append(AUC_Loss, ignore_index=True)
        AUC_Loss.to_csv(args.logdir + '/ChangeDataSize/' + net + '/AUC_Loss.csv')



def AdversaryTwo(args, Random_Data=False):
    if Random_Data:
        args.Split_Size = [[100], [2000], [100], [100]]
        img_sizes = [(3,32,32), (3,32,32), (3,64,64), (3, 128, 128)] 
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    num_class = args.num_classes[args.dataset_ID]
    
    logx.initialize(logdir=args.logdir + '/adversaryTwo', coolname=False, tensorboard=False)
    if args.blackadvattack == 'HopSkipJump':
        ITER = [50] # for call HSJA evaluation [1, 5, 10, 15, 20, 30]  default 50
    elif args.blackadvattack == 'QEBA':
        ITER = [150] # for call QEBA evaluation default 150
    elif args.blackadvattack == 'SaltandPepperNoise':
        ITER = [-1] # for call SaltandPepperNoise evaluation default 150
    for maxitr in ITER:
        AUC_Dist, Distance = [], []
        for cluster in split_size:
            torch.cuda.empty_cache()
            args.batch_size = 1
            if Random_Data:
                fake_set = datasets.FakeData(size=10000, image_size=img_sizes[args.dataset_ID], num_classes=num_class, transform= transforms.Compose([Rand_Augment(), transforms.ToTensor()]))
                data_loader = DataLoader(fake_set, batch_size=args.batch_size, shuffle=False)
            else:
                data_loader = load_dataset(args, dataset, cluster, mode='adversary', max_num=200)
            targetmodel = CNN('CNN7', dataset)
            targetmodel = nn.DataParallel(targetmodel.cuda())
            
            state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
            targetmodel.load_state_dict(state_dict)
            targetmodel.eval()
            
            if args.blackadvattack == 'HopSkipJump':
                AUC_Dist, Distance = AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data, maxitr)
            elif args.blackadvattack == 'QEBA':
                AUC_Dist, Distance = AdversaryTwo_QEBA(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data, maxitr)
            elif args.blackadvattack == 'SaltandPepperNoise':
                AUC_Dist, Distance = AdversaryTwo_SaltandPepperNoise(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data)

        df = pd.DataFrame()
        AUC_Dist = df.append(AUC_Dist, ignore_index=True)
        Distance = df.append(Distance, ignore_index=True)
        
        if Random_Data:
            AUC_Dist.to_csv(args.logdir + '/adversaryTwo/AUC_Dist_'+args.blackadvattack+'.csv')
            Distance.to_csv(args.logdir + '/adversaryTwo/Distance_Random_'+args.blackadvattack+'.csv')
        else:
            AUC_Dist.to_csv(args.logdir + '/adversaryTwo/AUC_Dist_'+args.blackadvattack + '.csv')
            Distance.to_csv(args.logdir + '/adversaryTwo/Distance_'+args.blackadvattack+'.csv')
        

def Decision_Radius(args):
    num_class = args.num_classes[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]

    for _, cluster in enumerate(split_size):
        torch.cuda.empty_cache()
        mem_set, non_set, transform = load_dataset(args, dataset, cluster, mode='radius')
        targetmodel = CNN('CNN7', dataset)

        targetmodel = nn.DataParallel(targetmodel.cuda())
        state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
        targetmodel.load_state_dict(state_dict)
        targetmodel.eval()

        logx.initialize(logdir=args.logdir + '/radius/' + str(cluster), coolname=False, tensorboard=False)

        max_num = 200 if 200 < len(mem_set) else len(mem_set)
        logx.msg('======================Starting Decision Radius Training Dataset ====================')
        certify(targetmodel, 'cuda', mem_set, transform, num_class,
                    mode='both', start_img=0, num_img=max_num, 
                    sigma=0.25, beta=16)

        logx.msg('======================Starting Decision Radius Testing Dataset ====================')
        certify(targetmodel, 'cuda', non_set, transform, num_class,
                mode='both', start_img=0, num_img=max_num, 
                sigma=0.25, beta=16)


##############################
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Decision-based Membership Inference Attack Toy Example') 
    parser.add_argument('--train', default=True, type=bool,
                        help='train or attack')
    parser.add_argument('--dataset_ID', default=False, type=int, 
                        help='CIFAR10=0, CIFAR100=1, GTSRB=2, Face=3')
    parser.add_argument('--datasets', nargs='+',
                        default=['CIFAR10', 'CIFAR100', 'GTSRB', 'Face'])
    parser.add_argument('--num_classes', nargs='+',
                        default=[10, 100, 43, 19])
    parser.add_argument('--Split-Size', nargs='+',
                        default=[[3000, 2000, 1500, 1000, 500, 100],                     #3000, 2000, 1500, 1000, 500, 100
                                [7000, 6000, 5000, 4000, 3000, 2000 ],                      #9000, 8000, 7000, 6000, 5000, 4000  # 7000, 6000, 5000, 4000, 3000, 2000
                                [600, 500, 400, 300, 200, 100  ],  #600, 500, 400, 300, 200, 100            
                                [350, 300, 250, 200, 150, 100  ],  #350, 300, 250, 200, 150, 100                
                                ]) 
    parser.add_argument('--batch-size', nargs='+', default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001 for adam; 0.1 for SGD)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', default=True,type=bool,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--blackadvattack', default='HopSkipJump', type=str,
                        help='adversaryTwo uses the adv attack the target Model: HopSkipJump; QEBA')
    parser.add_argument('--logdir', type=str, default='',
                        help='target log directory')
    parser.add_argument('--mode_type', type=str, default='',
                        help='the type of action referring to the load dataset')
    parser.add_argument('--advOne_metric', type=str, default='Loss_visual', help='AUC of Loss, Entropy, Maximum respectively; or Loss_visual')
    
    args = parser.parse_args()

    for dataset_idx in [0,1]:
        args.dataset_ID = dataset_idx
        args.logdir = 'results'+'/' + args.datasets[args.dataset_ID]
        action = 0
        # train
        if action == 0:
            args.mode_type = 'target'
            Train_Target_Model(args)
        elif action == 1:
            args.mode_type = 'shadow'
            Train_Shadow_Model(args)
        elif action == 2: 
            args.logdir = 'results/CIFAR100' 
            Train_Shadow_Model_ChangeDataSize(args)


        # attack
        elif action == 3:
            AdversaryOne(args)
        elif action == 4:    
            args.logdir = 'results/CIFAR100'  
            AdversaryOne_ChangeDataSize(args)
        elif action == 5:
            AdversaryTwo(args, Random_Data=False)

        # others
        elif action == 6:
            Decision_Radius(args)

if __name__ == "__main__":
    main()
