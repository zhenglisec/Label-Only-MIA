from PIL import Image
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps
import os
import shutil
import pandas as pd
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.legend

def load_dataset(args, dataset, cluster=None, mode = 'target', max_num = 2000):
    kwargs = {'num_workers': 2, 'pin_memory': True}
    # load trainset and testset
    
    if mode == 'shadow' or mode == 'ChangeDataSize':
        if dataset == 'GTSRB':
            transform = transforms.Compose([Rand_Augment(), transforms.Resize((64,64)), transforms.ToTensor()])
        else:
            transform = transforms.Compose([Rand_Augment(), transforms.ToTensor()])
    else:
        if dataset == 'GTSRB':
            transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset == 'CIFAR10':
        whole_set = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
        max_cluster = 3000
        test_size = 1000
    elif dataset == 'CIFAR100':
        whole_set = datasets.CIFAR100('../data', train=True, download=True, transform=transform)
        max_cluster = 7000
        test_size = 1000
    elif dataset == 'GTSRB':
        whole_set = datasets.ImageFolder('../data/GTSRB/', transform= transform)
        max_cluster = 600
        test_size = 500
    elif dataset == 'Face':
        whole_set = datasets.ImageFolder('../data/lfw/', transform=transform)
        max_cluster = 350
        test_size = 100
    elif dataset == 'TinyImageNet':
        whole_set = datasets.ImageFolder('../data/tiny-imagenet-200/train', transform=transform)
        max_cluster = 30000
        test_size = 2000
    length = len(whole_set)
    if mode == 'target':
        train_size = cluster
        remain_size = length - train_size - test_size
        train_set, _, test_set = dataset_split(whole_set, [train_size, remain_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'shadow': 
        train_size = length - max_cluster - test_size
        _, train_set, test_set = dataset_split(whole_set, [max_cluster, train_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader#, test_loader
    elif mode == 'salem': 
        train_size = length - max_cluster - test_size
        salme_train = int(train_size * 0.5)
        salme_test = train_size - salme_train
        _, train_set, test_set, _ = dataset_split(whole_set, [max_cluster, salme_train, salme_test, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'ChangeDataSize':
        train_size = cluster
        remain_size = length - max_cluster - train_size - test_size
        _, train_set, _, _ = dataset_split(whole_set, [max_cluster, train_size, remain_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader

    ###
    elif mode in ['adversaryOne', 'adversaryTwo',  'radius']:
        mem_size = min([cluster, test_size, max_num])
        non_size = mem_size
        remain_size = length - mem_size - non_size
        mem_set, _, non_set = dataset_split(whole_set, [mem_size, remain_size, non_size])

        data_set = ConcatDataset([mem_set, non_set])
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)
        if mode == 'radius':
            return mem_set, non_set, transform
        else:
            return data_loader
     



def load_dataset_old(args, dataset, cluster=None, mode = 'target', max_num = 1000):
    kwargs = {'num_workers': 2, 'pin_memory': True}
    # load trainset and testset
    
    if mode == 'shadow' or mode == 'ChangeDataSize':
        if dataset == 'GTSRB':
            transform = transforms.Compose([Rand_Augment(), transforms.Resize((64,64)), transforms.ToTensor()])
        else:
            transform = transforms.Compose([Rand_Augment(), transforms.ToTensor()])
    else:
        if dataset == 'GTSRB':
            transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset == 'CIFAR10':
        whole_set = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    elif dataset == 'CIFAR100':
        whole_set = datasets.CIFAR100('../data', train=True, download=True, transform=transform)
    elif dataset == 'GTSRB':
        whole_set = datasets.ImageFolder('../data/GTSRB/', transform= transform)
    elif dataset == 'Face':
        whole_set = datasets.ImageFolder('../data/lfw/', transform=transform)
    elif dataset == 'ImageNet':
        whole_set = datasets.ImageFolder('../data/tiny-imagenet-200/train', transform=transform)

    length = len(whole_set)
    quarter = int(length / 4)
    if mode == 'target':
        train_size = cluster
        test_size = 2 * quarter
        remain_size = length - train_size - test_size
        train_set, _, test_set = dataset_split(whole_set, [train_size, remain_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'shadow': 
        test_size = int(2 * quarter * 0.01) if dataset != 'Face' else 50
        train_size = 2*quarter - test_size
        remain_size = length - train_size - test_size
        _, train_set, test_set = dataset_split(whole_set, [remain_size, train_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'shadow_ImageNet': 
        train_size = 200000
        test_size = 10000
        remain_size = length - train_size - test_size
        train_set, test_set, _ = dataset_split(whole_set, [train_size, test_size, remain_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'ChangeDataSize':
        train_size = cluster
        test_size = 2 * quarter - cluster
        remain_size = length - train_size - test_size
        _, train_set, test_set = dataset_split(whole_set, [remain_size, train_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader

    ###
    elif mode in ['adversaryOne', 'adversaryTwo',  'radius']:
        if cluster <= max_num:
            mem_size = cluster
            non_size = cluster
            remain_size = length - mem_size - non_size
            mem_set, non_set, _ = dataset_split(whole_set, [mem_size, non_size, remain_size])
        else:
            mem_size = max_num
            non_size = max_num
            remain_size = length - 2 * cluster
            mem_set, _, non_set, _, _ = dataset_split(whole_set, [mem_size, cluster-mem_size, non_size, cluster-non_size, remain_size])
        data_set = ConcatDataset([mem_set, non_set])
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)
        if mode == 'radius':
            return mem_set, non_set, transform
        else:
            return data_loader
        '''
        if model_type == 'adversaryOne':
            return data_loader
        elif model_type == 'adversaryTwo':
            return data_set
        elif model_type == 'radius':
            transform = transform_gtsrb if dataset == 'GTSRB' else transform_normal
            return mem_set, non_set, transform
        '''

def dataset_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = list(range(sum(lengths)))
    np.random.seed(1)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class Rand_Augment():
    def __init__(self, Numbers=None, max_Magnitude=None):
        self.transforms = ['autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize',
                           'contrast', 'brightness', 'sharpness', 'shearX', 'shearY', 'translateX', 'translateY']
        if Numbers is None:
            self.Numbers = len(self.transforms) // 2
        else:
            self.Numbers = Numbers
        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        fillcolor = 128
        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation ,
            # it is no  need to obey  the value  in  autoaugment.py
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 0.2, 10),
            "translateY": np.linspace(0, 0.2, 10),
            "rotate": np.linspace(0, 360, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 231, 10),
            "contrast": np.linspace(0.0, 0.5, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.3, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,           
            "invert": [0] * 10
        }
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fill=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fill=fillcolor),
            "rotate": lambda img, magnitude: self.rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: img,
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def rand_augment(self):
        """Generate a set of distortions.
             Args:
             N: Number of augmentation transformations to apply sequentially. N  is len(transforms)/2  will be best
             M: Max_Magnitude for all the transformations. should be  <= self.max_Magnitude """

        M = np.random.randint(0, self.max_Magnitude, self.Numbers)

        sampled_ops = np.random.choice(self.transforms, self.Numbers)
        return [(op, Magnitude) for (op, Magnitude) in zip(sampled_ops, M)]

    def __call__(self, image):
        operations = self.rand_augment()
        for (op_name, M) in operations:
            operation = self.func[op_name]
            mag = self.ranges[op_name][M]
            image = operation(image, mag)
        return image

    def rotate_with_fill(self, img, magnitude):
        #  I  don't know why  rotate  must change to RGBA , it is  copy  from Autoaugment - pytorch
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

def save_code(args):
        os.makedirs(args.logdir + '/code', exist_ok=True)
        files=os.listdir(args.logdir + '/code',) 
        code_num = len([path for path in files if 'code' in path])
        os.makedirs(args.logdir + '/code' + '/code_'+str(code_num), exist_ok=True)
        shutil.copyfile('main.py', args.logdir + '/code' + '/code_'+str(code_num) + '/main.py')
        shutil.copyfile('deeplearning.py', args.logdir + '/code' + '/code_'+str(code_num) + '/deeplearning.py')
        shutil.copyfile('classifier.py', args.logdir + '/code' + '/code_'+str(code_num) + '/classifier.py')
        shutil.copyfile('utils.py', args.logdir + '/code' + '/code_'+str(code_num) + '/utils.py')
        shutil.copyfile('radius.py', args.logdir + '/code' + '/code_'+str(code_num) + '/radius.py')
        shutil.copyfile('attack.py', args.logdir + '/code' + '/code_'+str(code_num) + '/attack.py')
def fixed_seed(args):
    if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            cudnn.deterministic = True
##############################################
##############################################
##############################################
import torch.nn.init as init
import numpy as np

init_param = np.sqrt(2)
init_type = 'default'
def init_func(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            if init_type == 'normal':
                init.normal_(m.weight, 0.0, init_param)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=init_param)
            elif init_type == 'xavier_unif':
                init.xavier_uniform_(m.weight, gain=init_param)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
            elif init_type == 'kaiming_out':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_param)
            elif init_type == 'zero':
                init.zeros_(m.weight)
            elif init_type == 'one':
                init.ones_(m.weight)
            elif init_type == 'constant':
                init.constant_(m.weight, init_param)
            elif init_type == 'default':
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()
##############################################
##############################################
##############################################

from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt 
#from mpl_toolkits.mplot3d import Axes3D
def reduce_dimension_2d(data, target, flag=3, sort=True, cut_index=9, name='targettsne'):

    #data = data.view(data.size(0), -1).cpu().numpy()
    
    if sort:
        data = np.sort(data, axis=1)[:, 0:cut_index]

    #estimator = []
    if flag == 0:
        pca = PCA(n_components=2)
        estimator=pca.fit_transform(data)
        #estimator.append(pca)

    elif flag == 1:
        rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04)
        estimator=rbf_pca.fit_transform(data)

    elif flag == 2:
        tsvd = TruncatedSVD(n_components=2)
        estimator = tsvd.fit_transform(data)
        #estimator.append(tsvd)

    elif flag == 3:
        tsne = TSNE(n_components=2, init='pca')
        estimator = tsne.fit_transform(data)
        #estimator.append(tsne)
    
    '''
    ############################
    # create meshgrid
    resolution = 1000 # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(estimator[:,0])-1, np.max(estimator[:,0])+1
    X2d_ymin, X2d_ymax = np.min(estimator[:,1])-1, np.max(estimator[:,1])+1
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
    from sklearn.neighbors.classification import KNeighborsClassifier
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=30).fit(estimator, target) 
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))


    print(np.unique(voronoiBackground))

    ############################
    '''
    half = int(len(estimator)/2)
    Statistic_Data = []
    for idx, (x, y) in enumerate(estimator):
        if idx < half:
            xy = {'x':x, 'y':y, 'Target':target[idx], 'Status':'Member' }
            Statistic_Data.append(xy)
        elif idx < 2*half:
            xy = {'x':x, 'y':y, 'Target':target[idx], 'Status':'Non-member'}
            Statistic_Data.append(xy)
        #else:
            #xy = {'x':x, 'y':y, 'Target':target[idx], 'Status':'Random point'}
            #Statistic_Data.append(xy)
    df = pd.DataFrame()
    Statistic_Data = df.append(Statistic_Data, ignore_index=True)

    ############################
    # 设置ALL 还是 Part
    ALL_FLAG = True

    if ALL_FLAG:
        cat_data = Statistic_Data
    else:
        cat_data = Statistic_Data[(Statistic_Data.Target==3)|(Statistic_Data.Target==5)|(Statistic_Data.Target==7)]

    resolution = 1000 # 100x100 background pixels
    X2d_xmin, X2d_xmax = cat_data.min().x-3, cat_data.max().x+3
    X2d_ymin, X2d_ymax = cat_data.min().y-3, cat_data.max().y+3
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
    from sklearn.neighbors import KNeighborsClassifier
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    knn_data = cat_data[['x', 'y']].values
    knn_target = cat_data[['Target']].values
    background_model = KNeighborsClassifier(n_neighbors=10).fit(knn_data, knn_target.reshape(knn_target.size)) 
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    ############################
   

    large = 30 
    med = 16
    params = {  'figure.figsize': (9, 9),
                "font.size": med,     # 全局字号
                'font.family':'Times New Roman', # 全局字体
                "axes.spines.right":False,  # 坐标系-右侧线
                "axes.spines.top":False,   # 坐标系-上侧线
                "axes.spines.left":False,  # 坐标系-左侧线
                "axes.spines.bottom":False,   # 坐标系-下侧线
                'axes.titlesize': large,   # 坐标系-标题-字号
                'legend.fontsize': large,   # 图例-字号
                'axes.labelsize': med,   # 坐标系-标签-字号
                'axes.titlesize': med,   # 坐标系-标题-字号
                'xtick.labelsize': large,  # 刻度-标签-字号
                'ytick.labelsize': large,  # 刻度-标签-字号
                'figure.titlesize': large,
                "xtick.direction":'in',   # 刻度-方向
                "ytick.direction":'in'  # 刻度-方向
            }
    plt.rcParams.update(params)
    #plt.style.use('seaborn-darkgrid')
    #sns.set_style("darkgrid")

    # 读入数据
   
    # 添加图和坐标系
    fig, ax = plt.subplots() 


    # 设置颜色
    palette = [sns.xkcd_rgb['denim blue'], sns.xkcd_rgb['medium green']]
    point_palette = sns.color_palette('deep')
    #del point_palette[1]


    from matplotlib.pyplot import contourf

    if ALL_FLAG:
        ax.contourf(xx, yy, voronoiBackground, #colors=point_palette,
                                             alpha=1, levels=20) #10 for part/20 for all
    else:
        ax.contourf(xx, yy, voronoiBackground, colors=['#34618D', '#5EC962', '#1F978B'],# 删掉for all / 保留for part
                                             alpha=1, levels=10) #10 for part/20 for all
    #ax.contour(xx, yy, voronoiBackground, levels=10, linewidths=0.5,colors=['black']*10, linestyles=['--']*100)
    
    kwargs = {'edgecolor':"w", # for edge color
             'linewidth':0.6, # line width of spot
             'linestyle':'--', # line style of spot
                }
    markers = {'Member':'p', 'Non-member':'*'}
    ax = sns.scatterplot(x="x", y="y", 
                    hue="Status", style='Status',#hue_order=['Non-member', 'Member'],
                    palette=palette[0:2], markers=markers, s=[200, 150], #linewidth=[0,0], zorder=2
                    alpha= 0.8, data=cat_data, legend="full", ax=ax, **kwargs)
    '''
    for i in range(knn_data.shape[0]):
        plt.text(knn_data[i, 0], knn_data[i, 1], str(knn_target[i][0]),
                 color=plt.cm.Set1(knn_target[i][0] / 10.),
                 fontdict={'weight': 'bold', 'size': 13})
    '''
    #ax.set_xticklabels([''])
    #ax.set_yticklabels([''])
    #ax.set_ylabel('')
    #ax.set_xlabel('')
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    ax.axis('off')
    # 设置图例并且设置图例的字体及大小
    handles, labels = ax.get_legend_handles_labels()

    leg = ax.legend(handles[1:3], [ "Member", "Non-member"],
                        loc='upper left',  
                        frameon=True, framealpha=0.8, markerscale=3)
    '''
    leg = ax.legend(handles, [ ],
                        loc='lower left',  
                        frameon=False)
    '''
    for line in leg.get_lines():
        line.set_linewidth(5)
    #plt.legend(handles=None, labels=None)
    fig.tight_layout()
    if ALL_FLAG:
        plt.savefig('graph/adversary2/new_boundary/db.png')
        plt.savefig('graph/adversary2/new_boundary/db.pdf')
    else:
        plt.savefig('graph/adversary2/new_boundary/db_part.png')
        plt.savefig('graph/adversary2/new_boundary/db_part.pdf')
    plt.close()
    
'''
def save_distance(args, Overfitting, Distance, Statistic_Data):
    num = int(len(Distance)/2)
    for idx, distance in enumerate(Distance):   
        if idx < num:
            data = {'Overfitting Level':float(Overfitting), 'Linf Distance':distance, 'Status':'Member'}
        else:
            data = {'Overfitting Level':float(Overfitting), 'Linf Distance':distance, 'Status':'Non-member'}
        Statistic_Data.append(data)
    return Statistic_Data
'''
def reduce_dimension_2d_copy(data, target, flag=3, sort=True, cut_index=9, name='targettsne'):

    #data = data.view(data.size(0), -1).cpu().numpy()
    
    if sort:
        data = np.sort(data, axis=1)[:, 0:cut_index]

    #estimator = []
    if flag == 0:
        pca = PCA(n_components=2)
        estimator=pca.fit_transform(data)
        #estimator.append(pca)

    elif flag == 1:
        rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04)
        estimator=rbf_pca.fit_transform(data)

    elif flag == 2:
        tsvd = TruncatedSVD(n_components=2)
        estimator = tsvd.fit_transform(data)
        #estimator.append(tsvd)

    elif flag == 3:
        tsne = TSNE(n_components=2, init='pca')
        estimator = tsne.fit_transform(data)
        #estimator.append(tsne)
    
    '''
    ############################
    # create meshgrid
    resolution = 1000 # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(estimator[:,0])-1, np.max(estimator[:,0])+1
    X2d_ymin, X2d_ymax = np.min(estimator[:,1])-1, np.max(estimator[:,1])+1
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
    from sklearn.neighbors.classification import KNeighborsClassifier
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=30).fit(estimator, target) 
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))


    print(np.unique(voronoiBackground))

    ############################
    '''
    half = int(len(estimator)/2)
    Statistic_Data = []
    for idx, (x, y) in enumerate(estimator):
        if idx < half:
            xy = {'x':x, 'y':y, 'Target':target[idx], 'Status':'Member' }
            Statistic_Data.append(xy)
        elif idx < 2*half:
            xy = {'x':x, 'y':y, 'Target':target[idx], 'Status':'Non-member'}
            Statistic_Data.append(xy)
        #else:
            #xy = {'x':x, 'y':y, 'Target':target[idx], 'Status':'Random point'}
            #Statistic_Data.append(xy)
    df = pd.DataFrame()
    Statistic_Data = df.append(Statistic_Data, ignore_index=True)

    ############################
    # 设置ALL 还是 Part
    ALL_FLAG = False

    if ALL_FLAG:
        cat_data = Statistic_Data
    else:
        cat_data = Statistic_Data[(Statistic_Data.Target==3)|(Statistic_Data.Target==5)|(Statistic_Data.Target==7)]

    resolution = 1000 # 100x100 background pixels
    X2d_xmin, X2d_xmax = cat_data.min().x-1, cat_data.max().x+1
    X2d_ymin, X2d_ymax = cat_data.min().y-1, cat_data.max().y+1
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
    from sklearn.neighbors import KNeighborsClassifier
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    knn_data = cat_data[['x', 'y']].values
    knn_target = cat_data[['Target']].values
    background_model = KNeighborsClassifier(n_neighbors=10).fit(knn_data, knn_target.reshape(knn_target.size)) 
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    ############################
    
    # set color
    #current_palette = sns.color_palette('viridis')
    #del current_palette[1]
    #color = []
    #color.append(current_palette[0])
    #color.append(current_palette[1])
    f, ax = plt.subplots(figsize=(8, 8))
    #sns.despine(f, left=True, bottom=True)
    #, 'Random point':'o'}
    #alpha = {'Member':1, 'Non-member':0.5}

    from matplotlib.pyplot import contourf
    #fill_palette = plt.get_cmap('viridis')
    #print(fill_palette)
    if ALL_FLAG:
        ax.contourf(xx, yy, voronoiBackground, #colors=['#34618D', '#5EC962', '#1F978B'],# 删掉for all / 保留for part
                                             alpha=1, levels=20) #10 for part/20 for all
    else:
        ax.contourf(xx, yy, voronoiBackground, colors=['#34618D', '#5EC962', '#1F978B'],# 删掉for all / 保留for part
                                             alpha=1, levels=10) #10 for part/20 for all
    #ax.contour(xx, yy, voronoiBackground, levels=10, linewidths=0.5,colors=['black']*10, linestyles=['--']*100)

    point_palette = sns.color_palette('deep')
    #point_palette = sns.color_palette('Paired', 8)
    
    del point_palette[1]
  
    markers = {'Member':'s', 'Non-member':'X'}
    ax = sns.scatterplot(x="x", y="y",
                    hue="Status", style='Status',
                    palette=point_palette[0:2], markers=markers,
                    sizes=10, alpha=0.5, data=cat_data, legend="full", ax=ax)
    '''
    for i in range(knn_data.shape[0]):
        plt.text(knn_data[i, 0], knn_data[i, 1], str(knn_target[i][0]),
                 color=plt.cm.Set1(knn_target[i][0] / 10.),
                 fontdict={'weight': 'bold', 'size': 13})
'''

    #ax.set_xticklabels([''])
    #ax.set_yticklabels([''])
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    import matplotlib.legend as mlegend
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args(
                [ax])
    plt.legend(handles[1:3], labels[1:3], loc = 'upper left', fontsize=17)
    #plt.legend(handles=None, labels=None)
    plt.tight_layout()
    if ALL_FLAG:
        plt.savefig('graph/adversary2/new_boundary/db.png')
        plt.savefig('graph/adversary2/new_boundary/db.pdf')
    else:
        plt.savefig('graph/adversary2/new_boundary/db_part.png')
        plt.savefig('graph/adversary2/new_boundary/db_part.pdf')
    plt.close()
    
