from __future__ import print_function, division
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from argments import parser
import torch.nn.functional as F
from compute_ece import get_CE_metrics
from preprocess_data import ImageNetData
from vit import *
import logging
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from utils import set_seed
from tqdm import tqdm



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt = '%Y-%m-%d  %H:%M:%S %a'    
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''parameters'''

def knowledge_distillation_loss(student_logits, teacher_probs, temperature=1.0):
    student_probs = F.softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean') * (temperature**2)    
    return loss


def temperature_scaling(logits, temperature):
    return logits / temperature 



def train(model, dataloader, criterion, optimizer, scheduler):
    correct = 0
    total = 0
    total_loss = 0
    y_true_one_hot = []
    y_true = []
    y_pred = []
    labels_oneh = []
    preds_oneh = []
    model.train()
    count = 0
    
    for batch in tqdm(dataloader, desc=f"Training"):
        count += 1
        ids, images, labels, logits, att = batch
        
        images = torch.stack(images).to(device)
        labels = labels.to(device)
        logits = logits.to(device)
        #print("input", images.shape, labels.shape)#(64,3,224,224)
        optimizer.zero_grad()
        
        outputs, exp_loss = model(images, att, flag_train=True)
        
        _, predicted = torch.max(outputs.data, 1)
        
        probabilities = torch.softmax(outputs, dim=1)
        
        label_loss = criterion(outputs, labels)
        label_loss_mean = torch.mean(label_loss)
        
        y_one_hot = torch.nn.functional.one_hot(labels, num_classes=args.num_classes)#
        brier_score_loss = torch.mean((y_one_hot - outputs).pow(2))
        
        
        kd_loss = knowledge_distillation_loss(outputs, logits, args.temperature)
        

        total_model_loss = label_loss_mean + args.kd_weight * kd_loss + args.exp_weight * exp_loss + args.brier_weight * brier_score_loss 
        
        loss = total_model_loss
        
        loss.backward()
        optimizer.step()
        
        if count % 1000 == 0:   
            logger.info(f"ce_loss  = {label_loss_mean}")
            logger.info(f"kd_loss  = {kd_loss}")
            logger.info(f"exp_loss  = {exp_loss}")
            logger.info(f"brier_score_loss  = {brier_score_loss}")

        total_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        y_one_hot = torch.nn.functional.one_hot(labels, num_classes=args.num_classes)
        y_true_one_hot.extend(y_one_hot)
        
        y_pred.extend(probabilities.argmax(dim=1).cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        
        ###ece
        label_oneh = torch.nn.functional.one_hot(labels, num_classes=args.num_classes)
        label_oneh = label_oneh.cpu().detach().numpy()
        labels_oneh.extend(label_oneh)
        pred = probabilities.cpu().detach().numpy()
        preds_oneh.extend(pred)
        

    scheduler.step()
    current_lr = scheduler.get_lr()[0]
    accuracy = 100 * correct / total
    logger.info(f'Learning Rate: {current_lr:.6f}, Train_Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}%')

    
    preds_oneh = np.array(preds_oneh).flatten()
    labels_oneh = np.array(labels_oneh).flatten()
    ece, mce = get_CE_metrics(preds_oneh, labels_oneh)
    logger.info(f"ece = {ece:.4f}, mce= {mce:.4f}")
    
    return accuracy

    

def evaluate(model, dataloader, platt_scalers=None, flag_ood=False):
    model.eval()
    all_predictions = []
    all_labels = []
    y_pred = []
    y_true = []
    preds_oneh = []
    labels_oneh = []
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for ids, images, labels in tqdm(dataloader, desc=f"Deving"):
            images = torch.stack(images).to(device)
            labels = labels.to(device)
            outputs, attention = model(images)
            

            _, predicted = torch.max(outputs.data, 1)
            
            ce_loss = criterion(outputs, labels)
            loss = torch.mean(ce_loss)
            total_loss += ce_loss.item()
            new_probabilities = torch.softmax(outputs, dim=1)
            
            
            '''if flag_ood: 
                new_probabilities = []
                for platt_scaler in platt_scalers:
                    probab = platt_scaler.predict_proba(outputs.cpu().detach().numpy())
                    probab = probab[:, 1]
                    new_probabilities.append(probab)
                #logger.info(f"new_probab:{new_probabilities.shape}")
                new_probabilities = np.array(new_probabilities).squeeze(axis=1)
                new_probabilities = torch.tensor(new_probabilities).unsqueeze(0) 
                
                #logger.info(f"new_probabilities:{new_probabilities.shape}")
            else:
                new_probabilities = torch.softmax(outputs, dim=1)  
                #logger.info(f"new_probabilities1:{new_probabilities.shape}")'''
                
                 
            y_pred.extend(new_probabilities.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
                
            #####ece
            pred = new_probabilities.cpu().detach().numpy()
            preds_oneh.extend(pred)

            label_oneh = torch.nn.functional.one_hot(labels, num_classes=args.num_classes)
            label_oneh = label_oneh.cpu().detach().numpy()
            labels_oneh.extend(label_oneh)
            

    preds_oneh = np.array(preds_oneh).flatten()
    labels_oneh = np.array(labels_oneh).flatten()
    ece, mce = get_CE_metrics(preds_oneh, labels_oneh)
    logger.info(f"ece = {ece:.4f}, mce= {mce:.4f}")
   
    
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f'Test Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}')
        
    return accuracy


def save_pretrained(model, path):

    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'model_weights.bin'))


if __name__ == "__main__":
    args = parser()

    torch.cuda.set_device(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = max([4 * torch.cuda.device_count(), 4])

    set_seed(args.seed)
    
    logger = get_logger(f'./logs/{args.logging}.log')
    
    logger.info(f"Args:{args}")

  
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])


    train_dir = args.train_dir 
    logger.info(f"train_dir---: {train_dir}")
    iid_dir = "./data/i9/i9_original/val-wm-m"
    logger.info(f"iid_dir---: {iid_dir}")
    image_r_dir = "./data/i9/i9_shortcut/mixed_rand/val"
    logger.info(f"image_r_dir---: {image_r_dir}")
    image_r_wm_dir = "./data/i9/i9_shortcut/mixed_rand_wm"
    logger.info(f"image_r_wm_dir---: {image_r_wm_dir}")
    image_r_wm_rmbg_dir = "./data/i9/i9_shortcut/mixed_rand/val-rmbg-wm"
    logger.info(f"image_r_wm_rmbg_dir---: {image_r_wm_rmbg_dir}")
    
    
    train_IND1 = ImageNetData(args, train_dir) # [[id, image_path, label], ...]
    train_data = train_IND1.train_data
    train_dataloader = train_IND1.get_data_loader(train_data, args.batch_size, transform=train_transform, shuffle=True, flag_train=True)
    logger.info(f"len_train_data: {len(train_data)}")   
    logger.info(f"num_batches_train: {len(train_dataloader)}") 
    
    
    iid_IND = ImageNetData(args, iid_dir) # [[id, image_path, label], ...]
    iid_data = iid_IND.data
    iid_dataloader = iid_IND.get_data_loader(iid_data, batch_size=1, transform=val_transform, shuffle=False)
    logger.info(f"len_iid_data: {len(iid_data)}")   
    logger.info(f"num_batches_iid: {len(iid_dataloader)}") 

    image_r_IND = ImageNetData(args, image_r_dir) # [[id, image_path, label], ...]
    image_r_data = image_r_IND.data
    image_r_dataloader = image_r_IND.get_data_loader(image_r_data, batch_size=1, transform=val_transform, shuffle=True)
    logger.info(f"len_image_r_b: {len(image_r_data)}")   
    logger.info(f"num_batches_image_r_b: {len(image_r_dataloader)}")  


    image_r_wm_rmbg_IND = ImageNetData(args, image_r_wm_rmbg_dir) # [[id, image_path, label], ...]
    image_r_wm_rmbg_data = image_r_wm_rmbg_IND.data
    image_r_wm_rmbg_dataloader = image_r_wm_rmbg_IND.get_data_loader(image_r_wm_rmbg_data, batch_size=1, transform=val_transform, shuffle=False)
    logger.info(f"len_image_r_wm_: {len(image_r_wm_rmbg_data)}")   
    logger.info(f"num_batches_image_r_wm: {len(image_r_wm_rmbg_dataloader)}") 


    image_r_wm_IND = ImageNetData(args, image_r_wm_dir) # [[id, image_path, label], ...]
    image_r_wm_data = image_r_wm_IND.data
    image_r_wm_dataloader = image_r_wm_IND.get_data_loader(image_r_wm_data, batch_size=1, transform=val_transform, shuffle=False)
    logger.info(f"len_image_r_b+wm: {len(image_r_wm_data)}")   
    logger.info(f"num_batches_image_r_b+wm: {len(image_r_wm_dataloader)}") 
        
    
    model = VIT(args, device).to(device)
    logger.info(f"model:{model.named_parameters()}")


    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1)

    test_flag = False
    best_acc = 0.
    best_epoch = 0

    best_acc1 = 0.
    best_epoch1 = 0

    best_acc2 = 0.
    best_epoch2 = 0
    
    best_acc3 = 0.
    best_epoch3 = 0

    best_acc4 = 0.
    best_epoch4 = 0
    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

    for epoch in range(1, args.epoch_num + 1):
        logger.info(f"Epoch {epoch}:")
        train(model, train_dataloader, criterion, optimizer, scheduler)
    
        logger.info("iid:")
        acc_iid = evaluate(model, iid_dataloader)

        logger.info("mixed_rand_b:")
        acc_rand = evaluate(model, image_r_dataloader)

        logger.info("mixed_rand_w:")
        acc_rand_w = evaluate(model, image_r_wm_rmbg_dataloader)
        
        logger.info("mixed_rand_b+w:")
        acc_rand_b_w = evaluate(model, image_r_wm_dataloader)
        
        if acc_iid > best_acc2:
            best_acc2 = acc_iid
            best_epoch2 = epoch
            
            checkpoints_dirname = f"./model_outputs/{args.logging}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname+ 'iid/')
            
        if acc_rand > best_acc1:
            best_acc1 = acc_rand
            best_epoch1 = epoch
            
            checkpoints_dirname = f"./model_outputs/{args.logging}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname+ 'r/')
            
        if acc_rand_w > best_acc4:
            best_acc4 = acc_rand_w
            best_epoch4 = epoch
            
            checkpoints_dirname = f"./model_outputs/{args.logging}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname+ 'r_w/')
            
        
        if acc_rand_b_w > best_acc3:
            best_acc3 = acc_rand_b_w
            best_epoch3 = epoch
            
            checkpoints_dirname = f"./model_outputs/{args.logging}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname+ 'r_b_w/')
            
            
    logger.info(f"IID: Best Acc = {best_acc2}, epoch {best_epoch2}")           
    logger.info(f"mixed_same: Best Acc = {best_acc}, epoch {best_epoch}")     
    logger.info(f"mxied_rand_b: Best Acc = {best_acc1}, epoch {best_epoch1}")  
    logger.info(f"mxied_rand_w: Best Acc = {best_acc4}, epoch {best_epoch4}")  
    logger.info(f"mxied_rand_b+w: Best Acc = {best_acc3}, epoch {best_epoch3}")  


