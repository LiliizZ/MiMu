from __future__ import print_function, division
import math
import torch
import torch.nn as nn
from torchvision import transforms
from argments import parser
from preprocess import ImageNetData, get_data_loader

from vit import *

import logging
import os
import time

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



def adjust_learning_rate(args, optimizer, init_lr, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epoch_num))
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

def train(model, dataloader, criterion, optimizer, scheduler):

    correct = 0
    total = 0
    total_loss = 0
    y_true_one_hot = []
    y_true = []
    y_pred = []

    model.train()
    count = 0
    #current_lr = adjust_learning_rate(args, optimizer, args.lr, epoch)
    
    output_file_result = open(f"./results/{epoch}_train_output.txt", "a")
    att_file_result = open(f"./attention/{epoch}_train_output.txt", "a")
    for batch in tqdm(dataloader, desc=f"Training"):
        count += 1
        ids, images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        #print("input", images.shape, labels.shape)#(64,3,224,224)
        optimizer.zero_grad()
        
        outputs_m, attentions = model(images)
        _, predicted = torch.max(outputs_m.data, 1)
        probabilities = torch.softmax(outputs_m, dim=1)
        y_one_hot = torch.nn.functional.one_hot(labels, num_classes=9)
        brier_score_loss = torch.mean((y_one_hot - outputs_m).pow(2))
        

        label_loss = criterion(outputs_m, labels)
        label_loss_mean = torch.mean(label_loss)

        total_model_loss = label_loss_mean  + args.brier_weight * brier_score_loss 
        loss = total_model_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_one_hot = torch.nn.functional.one_hot(labels, num_classes=args.num_classes)
        y_true_one_hot.extend(y_one_hot)
        
        predds = probabilities.argmax(dim=1)
        y_pred.extend(predds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        
        if epoch == 0 or epoch == args.epoch_num:
            for id, loss, prob, y_one_hot, pre, y in zip(ids, label_loss, probabilities, y_one_hot, predds, labels):
                line = f"{id}\t{loss.cpu().detach().numpy()}\t{list(prob.cpu().detach().numpy())}\t{list(y_one_hot.cpu().detach().numpy())}\t\t{pre}\t{y}\t{True if pre==y else False}"  
                output_file_result.write(line+"\n")
    
            for id, att in zip(ids, attentions):
                line = f"{att}" 
                att_file_result.write(line+"\n")
            
    scheduler.step()
    current_lr = scheduler.get_lr()[0]
    accuracy = 100 * correct / total

    logger.info(f'Learning Rate: {current_lr:.6f}, Train_Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}%')


def evaluate(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for ids, images, labels in tqdm(dataloader, desc=f"Deving"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            ce_loss = criterion(outputs, labels)
            loss = torch.mean(ce_loss)
            total_loss += ce_loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = 100 * correct / total
    logger.info(f'Test Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}%')
        
    return accuracy

        
def save_pretrained(model, path):
    #save model
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

    train_dir = args.train_path
    logger.info(f"train----: {train_dir}")
    iid_dir = "/data/i9/i9_original/val-wm-m"   
    logger.info(f"iid----: {iid_dir}")
    image_r_dir = "/data/i9/i9_shortcut/mixed_rand/val"
    logger.info(f"mixed_rand----: {image_r_dir}")
    image_r_wm_dir = "/data/i9/i9_shortcut/mixed_rand_wm"
    logger.info(f"mixed_rand_wm----: {image_r_wm_dir}")
    image_r_wm_rmbg_dir = "/data/i9/i9_shortcut/mixed_rand/val-rmbg-wm"
    logger.info(f"image_r_wm_rmbg_dir---: {image_r_wm_rmbg_dir}")
    
    
    train_IND1 = ImageNetData(args, train_dir) # [[id, image_path, label], ...]
    train_data = train_IND1.data
    '''train_IND2 = ImageNetData(args, train_dir2)
    train_data2 = train_IND2.data
    train_data1.extend(train_data2)'''
    train_dataloader = get_data_loader(train_data, args.batch_size, transform=train_transform, shuffle=True)
    logger.info(f"len_train_data: {len(train_data)}")   
    logger.info(f"num_batches_train: {len(train_dataloader)}") 
    

    iid_IND = ImageNetData(args, iid_dir) # [[id, image_path, label], ...]
    iid_data = iid_IND.data
    iid_dataloader = get_data_loader(iid_data, batch_size=1, transform=val_transform, shuffle=False)
    logger.info(f"len_iid_data: {len(iid_data)}")   
    logger.info(f"num_batches_iid: {len(iid_dataloader)}") 


    image_r_IND = ImageNetData(args, image_r_dir) # [[id, image_path, label], ...]
    image_r_data = image_r_IND.data
    image_r_dataloader = get_data_loader(image_r_data, batch_size=1, transform=val_transform, shuffle=False)
    logger.info(f"len_image_r_data: {len(image_r_data)}")   
    logger.info(f"num_batches_image_r: {len(image_r_dataloader)}")  
    
    
    image_r_wm_IND = ImageNetData(args, image_r_wm_dir) # [[id, image_path, label], ...]
    image_r_wm_data = image_r_wm_IND.data
    image_r_wm_dataloader = get_data_loader(image_r_wm_data, batch_size=1, transform=val_transform, shuffle=False)
    logger.info(f"len_image_r_wm_data: {len(image_r_wm_data)}")   
    logger.info(f"num_batches_image_r_wm: {len(image_r_wm_dataloader)}")  
    
    image_r_wm_rmbg_IND = ImageNetData(args, image_r_wm_rmbg_dir) # [[id, image_path, label], ...]
    image_r_wm_rmbg_data = image_r_wm_rmbg_IND.data
    image_r_wm_rmbg_dataloader = get_data_loader(image_r_wm_rmbg_data, batch_size=1, transform=val_transform, shuffle=False)
    logger.info(f"len_image_r_wm_rmbg_data: {len(image_r_wm_rmbg_data)}")   
    logger.info(f"num_batches_image_r_wm_rmbg: {len(image_r_wm_rmbg_dataloader)}") 
    

    
    model = VIT(args, device).to(device) 

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
        logger.info("mixed_rand:")
        acc_rand = evaluate(model, image_r_dataloader)
        logger.info("mixed_rand_wm:")
        acc_rand_wm = evaluate(model, image_r_wm_dataloader)
        logger.info("mixed_rand_wm_rmbg:")
        acc_rand_wm_rmbg = evaluate(model, image_r_wm_rmbg_dataloader)

        if acc_iid > best_acc2:
            best_acc2 = acc_iid
            best_epoch2 = epoch
            
            checkpoints_dirname = f"./model_outputs/{args.logging}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname + 'iid_')
        
        if acc_rand > best_acc1:
            best_acc1 = acc_rand
            best_epoch1 = epoch
            
            checkpoints_dirname = f"./model_outputs/{args.logging}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname + 'r_')

        if acc_rand_wm > best_acc3:
            best_acc3 = acc_rand_wm
            best_epoch3 = epoch
            
            checkpoints_dirname = f"./model_outputs/{args.logging}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname + 'r_m_')
            
        if acc_rand_wm_rmbg > best_acc4:
            best_acc4 = acc_rand_wm_rmbg
            best_epoch4 = epoch
            
            checkpoints_dirname = f"./model_outputs/{args.logging}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname+ 'r_wm_rmbg/')
    
            
    logger.info(f"IID: Best Acc = {best_acc2}, epoch {best_epoch2}")           
    logger.info(f"mixed_same: Best Acc = {best_acc}, epoch {best_epoch}")     
    logger.info(f"mxied_rand_b: Best Acc = {best_acc1}, epoch {best_epoch1}") 
    logger.info(f"mxied_rand_w: Best Acc = {best_acc4}, epoch {best_epoch4}") 
    logger.info(f"mxied_rand_b+w: Best Acc = {best_acc3}, epoch {best_epoch3}")  
      
