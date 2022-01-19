import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from runx.logx import logx

from foolbox.distances import l0, l1, l2, linf
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success

import QEBA
from QEBA.criteria import TargetClass, Misclassification
from QEBA.pre_process.attack_setting import load_pgen

def prediction(x):
    x_list = x[0].tolist()
    x_sort = sorted(x_list)
    max_index = x_list.index(x_sort[-1])

    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum

    return softmax, max_index#, sec_index

def AdversaryOne_Feature(args, shadowmodel, data_loader, cluster, Statistic_Data):
    Loss = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.cuda()
            output = shadowmodel(data)
            Loss.append(F.cross_entropy(output, target.cuda()).item())
    Loss = np.asarray(Loss)
    half = int(len(Loss)/2)
    member = Loss[:half]
    non_member = Loss[half:]        
    for loss in member:
        Statistic_Data.append({'DataSize':float(cluster), 'Loss':loss,  'Status':'Member'})
    for loss in non_member:
        Statistic_Data.append({'DataSize':float(cluster), 'Loss':loss,  'Status':'Non-member'})
    return Statistic_Data


def AdversaryOne_evaluation(args, targetmodel, shadowmodel, data_loader, cluster, AUC_Loss, AUC_Entropy, AUC_Maximum):
    Loss = []
    Entropy = []
    Maximum = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            Toutput = targetmodel(data)
            Tlabel = Toutput.max(1)[1]

            Soutput = shadowmodel(data)
            if Tlabel != target:
               
                Loss.append(100)
            else:
                Loss.append(F.cross_entropy(Soutput, target).item())
            
            prob = F.softmax(Soutput, dim=1) 

            Maximum.append(torch.max(prob).item())
            entropy = -1 * torch.sum(torch.mul(prob, torch.log(prob)))
            if str(entropy.item()) == 'nan':
                Entropy.append(1e-100)
            else:
                Entropy.append(entropy.item())
 
    mem_groundtruth = np.ones(int(len(data_loader.dataset)/2))
    non_groundtruth = np.zeros(int(len(data_loader.dataset)/2))
    groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))

    predictions_Loss = np.asarray(Loss)
    predictions_Entropy = np.asarray(Entropy)
    predictions_Maximum = np.asarray(Maximum)
    
    fpr, tpr, _ = roc_curve(groundtruth, predictions_Loss, pos_label=0, drop_intermediate=False)
    AUC_Loss.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

    fpr, tpr, _ = roc_curve(groundtruth, predictions_Entropy, pos_label=0, drop_intermediate=False)
    AUC_Entropy.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

    fpr, tpr, _ = roc_curve(groundtruth, predictions_Maximum, pos_label=1, drop_intermediate=False)
    AUC_Maximum.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})
    return AUC_Loss, AUC_Entropy, AUC_Maximum
def AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data=False, maxitr=50, max_eval=10000):
    input_shape = [(3, 32, 32), (3, 32, 32), (3, 64, 64), (3, 128, 128)]
    nb_classes = [10, 100, 43, 19]
    ARTclassifier = PyTorchClassifier(
                model=targetmodel,
                clip_values=(0, 1),
                loss=F.cross_entropy,
                input_shape=input_shape[args.dataset_ID],
                nb_classes=nb_classes[args.dataset_ID],
            )
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    Attack = HopSkipJump(classifier=ARTclassifier, targeted =False, max_iter=maxitr, max_eval=max_eval)

    mid = int(len(data_loader.dataset)/2)
    member_groundtruth, non_member_groundtruth = [], []
    for idx, (data, target) in enumerate(data_loader): 
        targetmodel.module.query_num = 0
        data = np.array(data)  
        logit = ARTclassifier.predict(data)
        _, pred = prediction(logit)
        if pred != target.item() and not Random_Data:
            success = 1
            data_adv = data
        else:
            data_adv = Attack.generate(x=data) 
            data_adv = np.array(data_adv) 
            if Random_Data:
                success = compute_success(ARTclassifier, data, [pred], data_adv) 
            else:
                success = compute_success(ARTclassifier, data, [target.item()], data_adv)

        if success == 1:
            print(targetmodel.module.query_num)
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))

            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)

        if Random_Data and len(L0_dist)==100:
            break
        
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.asarray(L0_dist)
    L1_dist = np.asarray(L1_dist)
    L2_dist = np.asarray(L2_dist)
    Linf_dist = np.asarray(Linf_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L0_dist, pos_label=1, drop_intermediate=False)
    L0_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L1_dist, pos_label=1, drop_intermediate=False)
    L1_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)
    L2_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, Linf_dist, pos_label=1, drop_intermediate=False)
    Linf_auc = round(auc(fpr, tpr), 4)

    ### AUC based on distance
    auc_score = {'DataSize':float(cluster), 'L0_auc':L0_auc, 'L1_auc':L1_auc, 'L2_auc':L2_auc, 'Linf_auc':Linf_auc}
    AUC_Dist.append(auc_score)

    ### Distance of L0, L1, L2, Linf
    middle= int(len(L0_dist)/2)
    for idx, (l0_dist, l1_dist, l2_dist, linf_dist) in enumerate(zip(L0_dist, L1_dist, L2_dist, Linf_dist)):   
        if idx < middle:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Member'}
        else:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Non-member'}
        Distance.append(data)
    return AUC_Dist, Distance

def AdversaryTwo_QEBA(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data=False, max_iter=150):
    #input_shape = [(3, 32, 32), (3, 32, 32), (3, 64, 64), (3, 128, 128), (3, 64, 64)]
    nb_classes = [10, 100, 43, 19, 200]
    PGEN = ['resize768']
    p_gen, maxN, initN = load_pgen(args, PGEN[0])
    
    fmodel = QEBA.models.PyTorchModel(targetmodel, bounds=(0, 1), 
                num_classes=nb_classes[args.dataset_ID], discretize=False)
    Attack = QEBA.attacks.BAPP_custom(fmodel, criterion=Misclassification()) #criterion=TargetClass(src_label)
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    member_groundtruth, non_member_groundtruth = [], []
    mid = int(len(data_loader.dataset)/2)
    for idx, (data, target) in enumerate(data_loader):   
        targetmodel.module.query_num = 0
        data = data.numpy()
        data = np.squeeze(data)
        pred = np.argmax(fmodel.forward_one(data))
        if pred != target.item():
            data_adv = data
            pred_adv = pred
        else:
            grad_gt = fmodel.gradient_one(data, label=target.item())
            rho = p_gen.calc_rho(grad_gt, data).item()

            Adversarial = Attack(data, label=target.item(), starting_point = None, iterations=max_iter, stepsize_search='geometric_progression', 
                        unpack=False, max_num_evals=maxN, initial_num_evals=initN, internal_dtype=np.float32, 
                        rv_generator = p_gen, atk_level=999, mask=None, batch_size=1, rho_ref = rho, 
                        log_every_n_steps=1, suffix=PGEN[0], verbose=False)  

        
            data_adv = Adversarial.perturbed     
            pred_adv = Adversarial.adversarial_class

        if target.item() != pred_adv and type(data_adv) == np.ndarray:
            print(targetmodel.module.query_num)
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            data = data[np.newaxis, :]
            data_adv = data_adv[np.newaxis, :]
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))
            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)
        if Random_Data and len(L0_dist)==100:
            break
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.asarray(L0_dist)
    L1_dist = np.asarray(L1_dist)
    L2_dist = np.asarray(L2_dist)
    Linf_dist = np.asarray(Linf_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L0_dist, pos_label=1, drop_intermediate=False)
    L0_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L1_dist, pos_label=1, drop_intermediate=False)
    L1_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)
    L2_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, Linf_dist, pos_label=1, drop_intermediate=False)
    Linf_auc = round(auc(fpr, tpr), 4)

    ### AUC based on distance
    auc_score = {'DataSize':float(cluster), 'L0_auc':L0_auc, 'L1_auc':L1_auc, 'L2_auc':L2_auc, 'Linf_auc':Linf_auc}
    AUC_Dist.append(auc_score)

    ### Distance of L0, L1, L2, Linf
    middle= int(len(L0_dist)/2)
    for idx, (l0_dist, l1_dist, l2_dist, linf_dist) in enumerate(zip(L0_dist, L1_dist, L2_dist, Linf_dist)):   
        if idx < middle:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Member'}
        else:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Non-member'}
        Distance.append(data)
    return AUC_Dist, Distance
def AdversaryTwo_SaltandPepperNoise(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data=False, max_iter=150):
    nb_classes = [10, 100, 43, 19, 200]
    PGEN = ['resize768']
    # p_gen, maxN, initN = load_pgen(args, PGEN[0])
    
    fmodel = QEBA.models.PyTorchModel(targetmodel, bounds=(0, 1), 
                num_classes=nb_classes[args.dataset_ID], discretize=False)
    Attack = QEBA.attacks.SaltAndPepperNoiseAttack(fmodel, criterion=Misclassification()) #criterion=TargetClass(src_label)
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    member_groundtruth, non_member_groundtruth = [], []
    mid = int(len(data_loader.dataset)/2)
    for idx, (data, target) in enumerate(data_loader):   
        targetmodel.module.query_num = 0
        data = data.numpy()
        data = np.squeeze(data)
        pred = np.argmax(fmodel.forward_one(data))
   
        if pred != target.item():
            data_adv = data
            pred_adv = pred
        else:

            data_adv = Attack(data, label=target.item())  

            if type(data_adv) == np.ndarray:
                pred_adv = np.argmax(fmodel.forward_one(data_adv))
            else:
                continue
        if target.item() != pred_adv:
            print(targetmodel.module.query_num)
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            data = data[np.newaxis, :]
            data_adv = data_adv[np.newaxis, :]
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))
            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)
        if Random_Data and len(L0_dist)==100:
            break
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.asarray(L0_dist)
    L1_dist = np.asarray(L1_dist)
    L2_dist = np.asarray(L2_dist)
    Linf_dist = np.asarray(Linf_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L0_dist, pos_label=1, drop_intermediate=False)
    L0_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L1_dist, pos_label=1, drop_intermediate=False)
    L1_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)
    L2_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, Linf_dist, pos_label=1, drop_intermediate=False)
    Linf_auc = round(auc(fpr, tpr), 4)

    ### AUC based on distance
    auc_score = {'DataSize':float(cluster), 'L0_auc':L0_auc, 'L1_auc':L1_auc, 'L2_auc':L2_auc, 'Linf_auc':Linf_auc}
    AUC_Dist.append(auc_score)

    ### Distance of L0, L1, L2, Linf
    middle= int(len(L0_dist)/2)
    for idx, (l0_dist, l1_dist, l2_dist, linf_dist) in enumerate(zip(L0_dist, L1_dist, L2_dist, Linf_dist)):   
        if idx < middle:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Member'}
        else:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Non-member'}
        Distance.append(data)
    return AUC_Dist, Distance

