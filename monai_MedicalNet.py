
import logging
import os
import sys
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from sklearn.impute import SimpleImputer
# from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.config import print_config
from monai.data import decollate_batch
from monai.data import DataLoader, ImageDataset
from monai.utils import set_determinism
from monai.metrics import ROCAUCMetric
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ConcatItemsd,
    ScaleIntensityd,
    Resized,
    RandRotate90d,
    RandFlipd,
    ToTensord,
    Activations,
    AsDiscrete
)

import wandb

sys.path.append('/home/claire/Radioval/MedicalNet')
import setting
from model import generate_model
from datasets.brains18 import BrainS18Dataset




if __name__ == '__main__':

    check = False
    
    config={
        'experiment_name' : 'test_1',
        'dir_experiment' : '/mnt/sdb/datasets/DL_Monai/experiments',
        'dir_data' : '/mnt/sdb/datasets/dataset_nifti/cropped_mris_15cm',
        'pretrained_model' : '/home/claire/Radioval/MedicalNet/pretrain/resnet_50.pth',
        'model_depth' : 50,
        'resnet_shortcut' : 'B',
        'num_workers' : 2,
        'batch_size' : 2,
        'learning_rate': 1e-3,
        'epochs': 50,
        'image_size' : 150,
        'images_labels' : '/mnt/sdb/datasets/dataset_nifti/cropped_mris_15cm/path_images_labels/subtracted_images_masks_labels.csv',
        'train_val_test_splits' : '/mnt/sdb/datasets/dataset_nifti/cropped_mris_15cm/splits_finals/BCN_MRI_train_val_test_splits.csv'
        }

    run = wandb.init(project="MedicalNet-resnet50", config=config)

    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if pin_memory else "cpu")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    set_determinism(seed=0)

    dir_experiment = wandb.config.get("dir_experiment")
    experiment_name = wandb.config.get("experiment_name")

    output_folder = os.path.join(dir_experiment, experiment_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images_labels_csv = wandb.config.get("images_labels")
    train_val_test_splits_csv = wandb.config.get("train_val_test_splits")

    images_labels_df = pd.read_csv(images_labels_csv)
    train_val_test_splits_df = pd.read_csv(train_val_test_splits_csv)

    train_patient_ids = train_val_test_splits_df['train_split'].to_list()
    val_patient_ids = train_val_test_splits_df['val_split'].to_list()
    test_patient_ids = train_val_test_splits_df['test_split'].to_list()

    train_images = images_labels_df[images_labels_df['patient_id'].isin(train_patient_ids)]['path_to_images'].to_list()
    train_masks = images_labels_df[images_labels_df['patient_id'].isin(train_patient_ids)]['path_to_masks'].to_list()
    train_labels = images_labels_df[images_labels_df['patient_id'].isin(train_patient_ids)]['pcr'].fillna(0)
    val_images = images_labels_df[images_labels_df['patient_id'].isin(val_patient_ids)]['path_to_images'].to_list()
    val_masks = images_labels_df[images_labels_df['patient_id'].isin(val_patient_ids)]['path_to_masks'].to_list()
    val_labels = images_labels_df[images_labels_df['patient_id'].isin(val_patient_ids)]['pcr'].fillna(0)
    test_images = images_labels_df[images_labels_df['patient_id'].isin(test_patient_ids)]['path_to_images'].to_list()
    test_masks = images_labels_df[images_labels_df['patient_id'].isin(test_patient_ids)]['path_to_masks'].to_list()
    test_labels = images_labels_df[images_labels_df['patient_id'].isin(test_patient_ids)]['pcr'].fillna(0)


    with open(os.path.join(output_folder, 'train.txt'), 'w') as f:
        for train_image, train_mask, train_label in zip(train_images, train_masks, train_labels):
            f.write(f"{train_image} {train_mask} {int(train_label)}\n")
    with open(os.path.join(output_folder, 'val.txt'), 'w') as f:
        for val_image, val_mask, val_label in zip(val_images, val_masks, val_labels):
            f.write(f"{val_image} {val_mask} {int(val_label)}\n")
    with open(os.path.join(output_folder, 'test.txt'), 'w') as f:
        for test_image, test_mask, test_label in zip(test_images, test_masks, test_labels):
            f.write(f"{test_image} {test_mask} {int(test_label)}\n")
    
    # train_labels = torch.nn.functional.one_hot(torch.as_tensor(np.array(train_labels.to_list(), dtype=np.int64))).float()
    # val_labels = torch.nn.functional.one_hot(torch.as_tensor(np.array(val_labels.to_list(), dtype=np.int64))).float()
    # test_labels = torch.nn.functional.one_hot(torch.as_tensor(np.array(test_labels.to_list(), dtype=np.int64))).float()

    # # Define transforms
    # train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((image_size, image_size, image_size)), RandRotate90()])
    # val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((image_size, image_size, image_size))])

    # # create a training data loader
    # train_ds = ImageDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    # # create a validation data loader
    # val_ds = ImageDataset(image_files=val_images, labels=val_labels, transform=val_transforms)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    # # create a test data loader
    # test_ds = ImageDataset(image_files=test_images, labels=test_labels, transform=val_transforms)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2, pretrained=True).to(device)
            
    sets = setting.parse_opts() 

    sets.img_list = os.path.join(output_folder, 'train.txt')
    if check:
        sets.img_list = os.path.join(output_folder, 'train_check.txt')
    sets.n_epochs = wandb.config.get("epochs")
    if pin_memory:
        sets.no_cuda = False
        sets.gpu_id = "0"
        sets.pin_memory = True
    else:
        sets.no_cuda = True
    sets.learning_rate = wandb.config.get('learning_rate')
    sets.data_root = wandb.config.get('dir_data')
    sets.pretrain_path = wandb.config.get('pretrained_model')
    sets.num_workers = wandb.config.get('num_workers')
    sets.model_depth = wandb.config.get('model_depth')
    sets.resnet_shortcut = wandb.config.get('resnet_shortcut')
    sets.input_D = wandb.config.get('image_size')
    sets.input_H = wandb.config.get('image_size')
    sets.input_W = wandb.config.get('image_size')
       
    # getting model
    model, parameters = generate_model(sets) 
    model.to(device)
    print(model)
    # optimizer
    params = [
            { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
            { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
            ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # if sets.resume_path:
    
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
    batch_size = wandb.config.get('batch_size')
    training_dataset = BrainS18Dataset(sets.data_root, sets.img_list, sets)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=pin_memory)

    img_list_val = os.path.join(output_folder, 'val.txt')
    if check:
        img_list_val = os.path.join(output_folder, 'val_check.txt')
    val_dataset = BrainS18Dataset(sets.data_root, img_list_val, sets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=pin_memory)
    # training
    max_epochs = wandb.config.get('epochs')
    batches_per_epoch = len(train_loader)
    print('{} epochs in total, {} batches per epoch'.format(max_epochs, batches_per_epoch))
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    auc_metric = ROCAUCMetric()

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_function = loss_function.cuda()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, _, labels = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
            # inputs, _, labels = batch_data[0], batch_data[1], batch_data[2]
            if not sets.no_cuda: 
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(training_dataset) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, _, val_labels = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                # y_b = decollate_batch(y, detach=False)
                # y_pred_b = decollate_batch(y_pred)
                auc_metric(y_pred[:,1], y)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                # del y_pred_b, y_b
                metric_values.append(auc_result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(output_folder, "fine_tuning_resnet50.pth"))
                    print("saved new best metric model")
                print(f"Epoch: {epoch + 1} current AUC: {auc_result:.4f}"
                        f" Current accuracy: {acc_metric:.4f}"
                        f" Best AUC: {best_metric:.4f}"
                        f" at epoch : {best_metric_epoch}")
                wandb.log({"epoch": epoch + 1, "AUC": auc_result, "accuracy": acc_metric, "best AUC": best_metric, "best epoch": best_metric_epoch})
            
    run.log_model(path=os.path.join(output_folder, "best_metric_model_classification3d_dict.pth"), name="fine_tuning_resnet50")
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    wandb.log({"best accuracy": best_metric, "best epoch": best_metric_epoch})

    # Plot the loss and metric
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    wandb.log({"Epoch Average Loss": plt})
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    wandb.log({"Val AUC": plt})
    plt.savefig(os.path.join(output_folder, "training_loss_metric.png"))
    plt.close()
    wandb.finish()
          

    # for epoch in range(max_epochs):
    #     print("-" * 10)
    #     print(f"epoch {epoch + 1}/{max_epochs}")
    #     model.train()
    #     epoch_loss = 0
    #     step = 0

    #     for batch_data in train_loader:
    #         step += 1
    #         inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = loss_function(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #         epoch_len = len(train_ds) // train_loader.batch_size
    #         print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    #         # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            

    #     epoch_loss /= step
    #     epoch_loss_values.append(epoch_loss)
    #     print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    #     wandb.log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    #     if (epoch + 1) % val_interval == 0:
    #         model.eval()

    #         num_correct = 0.0
    #         metric_count = 0
    #         for val_data in val_loader:
    #             val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
    #             with torch.no_grad():
    #                 val_outputs = model(val_images)
    #                 value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
    #                 metric_count += len(value)
    #                 num_correct += value.sum().item()

    #         metric = num_correct / metric_count
    #         metric_values.append(metric)

    #         if metric > best_metric:
    #             best_metric = metric
    #             best_metric_epoch = epoch + 1
    #             torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
    #             print("saved new best metric model")

    #         print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
    #         print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
    #         # writer.add_scalar("val_accuracy", metric, epoch + 1)
    #         wandb.log(f"Val accuracy: {metric:.4f} at epoch {epoch + 1}")
            

    # print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    # wandb.log(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
    # wandb.finish()
