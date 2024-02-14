from setting import parse_opts 
# from datasets.brains18 import BrainS18Dataset
from datasets.bcnmri import BreastMRIDataset
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os
# import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss, accuracy_score
import wandb

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_prob = []
    y_pred = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_id, batch_data in enumerate(dataloader):
            volumes, class_array = batch_data
            # volumes, mask_array, class_array = batch_data
            volumes, class_array = volumes.to(device), class_array.to(device)
            outputs = model(volumes)
            # probabilities = torch.softmax(outputs, dim=1) # for multiclass
            # probabilities = torch.sigmoid(outputs, dim=1) # for binary
            # probabilities = outputs # our output it's already a sigmoid
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            # predicted = (outputs > 0.5).float() 
            
            correct += (predicted == class_array.reshape(-1)).sum().item()
            # correct += (predicted == class_array[:,0]).sum().item()
            total += class_array.size(0)
            # y_true.extend(class_array[:,0].cpu().numpy())

            y_true.extend(list(class_array.reshape(-1).cpu().numpy()))
            y_prob.extend(list(probabilities[:,1].cpu().numpy()))
            y_pred.extend(list(predicted.cpu().numpy()))

    # accuracy = correct / total
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier_score = brier_score_loss(y_true, y_prob)
    
    return accuracy, balanced_accuracy, roc_auc, brier_score

def train(data_loader, val_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))

    # loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    loss_seg = nn.BCEWithLogitsLoss()

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()
    
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        model.train()
        log.info('Start epoch {}'.format(epoch))
        log.info('lr = {}'.format(scheduler.get_last_lr()))
        running_loss = 0.0
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks, class_array = batch_data ##### volumes, label_masks = batch_data

            if not sets.no_cuda: 
                volumes = volumes.cuda()
                class_array = class_array.cuda() #####

            optimizer.zero_grad()
            out_masks = model(volumes)
            # print(out_masks.shape)
            # resize label
            # [n, _, d, h, w] = out_masks.shape
            # new_label_masks = np.zeros([n, d, h, w])
            # for label_id in range(n):
            #     label_mask = label_masks[label_id]
            #     [ori_c, ori_d, ori_h, ori_w] = label_mask.shape 
            #     label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
            #     scale = [d*1.0/ori_d, h*1.0/ori_h, w*1.0/ori_w]
            #     label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
            #     new_label_masks[label_id] = label_mask
            # new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            # if not sets.no_cuda:
            #     new_label_masks = new_label_masks.cuda()

            # calculating loss
            # labels = class_array.to('cuda')
            out_masks = out_masks.to('cuda')
            # loss_value_seg = loss_seg(out_masks, labels) ##### loss_value_seg = loss_seg(out_masks, new_label_masks)
            # loss_value_seg = loss_seg(out_masks[:,0], labels[:,0])
            labels_one_hot = np.eye(2)[class_array.cpu().numpy().astype(int)]
            loss_value_seg = loss_seg(out_masks, torch.from_numpy(labels_one_hot.squeeze()).cuda())
            loss = loss_value_seg
            # loss.requires_grad_(True) #####
            loss.backward()                
            optimizer.step()

            running_loss += loss.item()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batches_per_epoch, loss.item(), avg_batch_time))
            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    
                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                    torch.save({
                                'ecpoch': epoch,
                                'batch_id': batch_id,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                model_save_path)

        epoch_loss = running_loss / batches_per_epoch
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")
        if sets.enable_wandb:
            wandb.log({"epoch": epoch + 1, "loss": epoch_loss, "lr_core": scheduler.get_last_lr()[0], "lr_last": scheduler.get_last_lr()[1]})

        # Evaluate model on validation data
        val_accuracy, val_balanced_accuracy, val_roc_auc, val_brier_score = evaluate_model(model, val_loader, 'cuda')
        print("Validation Accuracy:", val_accuracy)
        print("Validation Balanced Accuracy:", val_balanced_accuracy)
        print("Validation ROC AUC:", val_roc_auc)
        print("Validation Brier Score:", val_brier_score)

        if sets.enable_wandb:
            wandb.log({"Acc_val": val_accuracy, "BAcc_val": val_balanced_accuracy, 
                    "ROC-AUC_val": val_roc_auc, "Brier_score_val": val_balanced_accuracy})
        
        scheduler.step()
        # # Update learning rate scheduler based on balanced accuracy
        # scheduler.step(val_balanced_accuracy)

                                    
    print('Finished training')            
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    if sets.ci_test:
        sets.img_list = './toy_data/test_ci.txt' 
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 14
        sets.input_H = 28
        sets.input_W = 28
    
    ## Experiment settings
    sets.img_list = '/home/lidia/source/MedicalNet/datasets/train.txt'
    sets.img_list_val = '/home/lidia/source/MedicalNet/datasets/val.txt'
    # sets.img_list = '/home/lidia/source/MedicalNet/datasets/train_check.txt'
    # sets.img_list_val = '/home/lidia/source/MedicalNet/datasets/val_check.txt'
    # sets.img_list_test = '/home/lidia/source/MedicalNet/datasets/test.txt'
    sets.data_root = '/datasets/BCN-MRI/claire/cropped_mris_15cm'
    # sets.gpu_id = [0]
    sets.input_D = 150
    sets.input_H = 150
    sets.input_W = 150
    sets.resume_path = ''
    sets.learning_rate = 0.000001 # set to 0.001 when finetune
    sets.batch_size = 6
    sets.num_workers = 4
    sets.n_epochs = 200
    sets.save_intervals = 5 # every 5 epochs
    sets.n_seg_classes = 2
    sets.model = 'resnet'
    # sets.pretrain_path = '/home/lidia/source/MedicalNet/pretrain/resnet_50_23dataset.pth'
    # sets.model_depth = 50
    # sets.resnet_shortcut = 'B'
    # sets.pretrain_path = '/home/lidia/source/MedicalNet/pretrain/resnet_34_23dataset.pth'
    # sets.model_depth = 34
    # sets.resnet_shortcut = 'A'
    sets.pretrain_path = '/home/lidia/source/MedicalNet/pretrain/resnet_18_23dataset.pth'
    sets.model_depth = 18
    sets.resnet_shortcut = 'A'
    # sets.pretrain_path = '/home/lidia/source/MedicalNet/pretrain/resnet_10_23dataset.pth'
    # sets.pretrain_path = '/home/lidia/source/MedicalNet/pretrain/resnet_10.pth'
    # sets.model_depth = 10
    # sets.resnet_shortcut = 'B'
    sets.enable_wandb = True

    os.environ["NUMEXPR_MAX_THREADS"]=str(8)

    experiment_name = 'resnet_18_23_one_hot_1'
    sets.save_folder = "/home/lidia/source/MedicalNet/experiments/{}".format(experiment_name)

    if sets.enable_wandb:
        # 1. Start a W&B run
        wandb.init(name=experiment_name, project="pcr-classification", entity='bcnaim', config=sets.__dict__)
        # # 2. Save model inputs and hyperparameters
        # config = wandb.config
        # config.learning_rate = 1e-3

        # getting model
        torch.manual_seed(sets.manual_seed)

    model, parameters = generate_model(sets)
    print(model)
    # model = MedicalNet(path_to_weights="pretrain/resnet_50.pth", device='cuda') #####
    # model.cuda()
    # for param_name, param in model.named_parameters():
    #     if param_name.startswith("conv_seg"):
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100}
                ]
        
    # optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.AdamW(params, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    

    sets.phase = 'train'
    training_dataset = BreastMRIDataset(sets.data_root, sets.img_list, sets)
    data_loader_train = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True,
                                    num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    sets.phase = 'val'
    val_dataset = BreastMRIDataset(sets.data_root, sets.img_list_val, sets)
    data_loader_val = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                  num_workers=1, pin_memory=sets.pin_memory)

    # training
    train(data_loader_train, data_loader_val, model, optimizer, scheduler, total_epochs=sets.n_epochs,
           save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 
    
