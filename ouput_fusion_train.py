import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations,AddChanneld,AsDiscrete,Compose,LoadImaged,RandRotate90d,Resized,ScaleIntensityd,EnsureTyped,EnsureType,Spacingd
from networks.feature_fusion import FF

import pandas as pd

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout,level=logging.INFO)
    ct=pd.read_csv(r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/ct-feature.csv')
    ct_feature_all=np.array(ct) #[533,1041]
    ct_feature=ct_feature_all[:,1:]
    ct_feature=np.array(ct_feature,dtype=np.float)

    bw = pd.read_csv(r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/bw-feature.csv')
    bw_feature_all = np.array(bw)  # [3209,1025]
    bw_feature = bw_feature_all[:, 1:]
    bw_feature = np.array(bw_feature, dtype=np.float)

    xx = pd.read_excel(r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/ROIs.xlsx',
                       sheet_name='ROIs_mask_label')
    labels = xx.iloc[:, 3].tolist()  # all labels

    train_files=[{"label":label,"ct":ct,"bw":bw} for label,ct,bw in zip(labels[:2258],ct_feature[:2258,:],bw_feature[:2258,:])] #train images,labels
    val_files=[{"label":label, "ct":ct,"bw":bw} for label,ct,bw in zip(labels[2258:],ct_feature[2258:,:],bw_feature[2258:,:])]

    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files)
    check_loader = DataLoader(check_ds, batch_size=500, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["ct"], check_data["bw"],check_data["label"])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files)
    train_loader = DataLoader(train_ds, batch_size=500, shuffle=True, num_workers=4,
                              pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files)
    val_loader = DataLoader(val_ds, batch_size=951, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model=monai.networks.nets.DenseNet121(spatial_dims=3,in_channels=1,out_channels=2).to(device)
    model = FF().to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    auc_metric = ROCAUCMetric

    # start a typical pytorch training
    max_epoch = 500
    val_interval = 50
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()

    for epoch in range(max_epoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epoch}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            ct,bw, labels = batch_data["ct"].to(device),batch_data["bw"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(ct,bw)  # insert clinic data
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{epoch + 1}/{max_epoch}, {step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_ct,val_bw, val_labels = val_data["ct"].to(device), val_data["bw"].to(device),val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_ct,val_bw)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                # y_onehot=[post_label(i) for i in decollate_batch(y)]#?  calculation AUC?
                # y_pred_act=[post_pred(i) for i in decollate_batch(y_pred)]
                # auc_metric(y_pred_act,y_onehot)
                # auc_result=acc_metric.aggregrate()
                # acc_metric.reset()
                # del y_pred_act,y_onehot
                torch.save(model.state_dict(),
                           "C:/YangLi/JHU-PET-CT/data20221013/PSMA_CLASSIFICATION_DATA/Trained model-fuse output-feature Transformer/classification3d_dict" + str(
                               epoch + 1) + ".pth")
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(),
                               "C:/YangLi/JHU-PET-CT/data20221013/PSMA_CLASSIFICATION_DATA/Trained model-fuse output-feature Transformer/best_metric_classification3d_dict.pth")
                    print("saved new best metric model")
                print("current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, acc_metric, best_metric, best_metric_epoch))
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()

