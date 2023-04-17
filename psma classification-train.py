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
# from networks.densenet_dualimage_path import DenseNet
# from networks.densenet import DenseNet  #single path
from networks.densenet_fuse_input import DenseNet  #fuse the input with 3D conv

import pandas as pd


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout,level=logging.INFO)
    ct_path=r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/CT/'
    bw_path=r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/SUVbw/'
    xx=pd.read_excel(r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/ROIs.xlsx',sheet_name='ROIs_mask_label')
    images_ct=xx.iloc[:,4].tolist()#all ct images
    images_bw = xx.iloc[:, 5].tolist()  # all suvbw images
    lab = xx.iloc[:, 3].tolist() #all labels


    labels_all=np.array(lab,dtype=np.int64)

    ct=images_ct[0:3209]  #train+val images
    bw=images_bw[0:3209]
    labels=labels_all[0:3209] #train+val labels


    ct1=[os.sep.join([ct_path,f]) for f in ct]
    bw1=[os.sep.join([bw_path,f]) for f in bw]

    train_files=[{"ct":ct2+'.nii.gz',"bw":bw2+'.nii.gz', "label":label} for ct2,bw2,label in zip(ct1[:2258], bw1[:2258], labels[:2258]) ] #train images,labels
    val_files=[{"ct":ct2+'.nii.gz',"bw":bw2+'.nii.gz', "label":label} for ct2,bw2,label in zip(ct1[2258:], bw1[2258:], labels[2258:]) ] #val images,labels

    train_transforms=Compose([LoadImaged(keys=["ct","bw"]),
                              AddChanneld(keys=["ct","bw"]),
                              ScaleIntensityd(keys=["ct","bw"]),
                              Spacingd(keys=["ct","bw"],pixdim=(1, 1, 1),mode=("bilinear")),
                              Resized(keys=["ct","bw"],spatial_size=(48,48,48)),
                              RandRotate90d(keys=["ct","bw"],prob=0.8,spatial_axes=[0,2]),
                              EnsureTyped(keys=["ct","bw"])])

    val_transforms=Compose([LoadImaged(keys=["ct","bw"]),
                              AddChanneld(keys=["ct","bw"]),
                              ScaleIntensityd(keys=["ct","bw"]),
                              Spacingd(keys=["ct","bw"], pixdim=(1, 1, 1), mode=("bilinear")),
                              Resized(keys=["ct","bw"],spatial_size=(48,48,48)),
                              RandRotate90d(keys=["ct","bw"],prob=0.8,spatial_axes=[0,2]),
                              EnsureTyped(keys=["ct","bw"])])

    post_pred=Compose([EnsureType(),Activations(softmax=True)])
    post_label=Compose([EnsureType(),AsDiscrete(to_onehot=2)])

    # Define dataset, data loader
    check_ds=monai.data.Dataset(data=train_files,transform=train_transforms)
    check_loader=DataLoader(check_ds,batch_size=10,num_workers=4,pin_memory=torch.cuda.is_available())
    check_data=monai.utils.misc.first(check_loader)
    print(check_data["ct"].shape,check_data["bw"].shape,check_data["label"])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=10, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

   #  model=DenseNet(spatial_dims=3,in_channels_ini=1,out_channels=2,init_features=64,
   #                 growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, act=('relu', {'inplace': True}),
   #                 norm='batch', dropout_prob=0.0).to(device)
    model = DenseNet(spatial_dims=3, in_channels=2, out_channels=2, init_features=64,
                     growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, act=('relu', {'inplace': True}),
                     norm='batch', dropout_prob=0.0).to(device)
    loss_function=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),1e-5)
    auc_metric=ROCAUCMetric

    # start a typical pytorch training
    max_epoch=1000
    val_interval=50
    best_metric=-1
    best_metric_epoch=-1
    writer=SummaryWriter()

    for epoch in range(max_epoch):
        print("-"*10)
        print(f"epoch {epoch+1}/{max_epoch}")
        model.train()
        epoch_loss=0
        step=0
        for batch_data in train_loader:
            step+=1
            ct_in,bw_in,labels=batch_data["ct"].to(device),batch_data["bw"].to(device),batch_data["label"].to(device)
            optimizer.zero_grad()
            x_ct1=model(ct_in,bw_in) # insert clinic data
            loss=loss_function(x_ct1,labels)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            epoch_len=len(train_ds)//train_loader.batch_size
            print(f"{epoch+1}/{max_epoch}, {step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss",loss.item(),epoch_len*epoch+step)
        epoch_loss/=step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch+1)%val_interval==0:
            model.eval()
            with torch.no_grad():
                y_pred=torch.tensor([],dtype=torch.float32,device=device)
                y=torch.tensor([],dtype=torch.long,device=device)
                for val_data in val_loader:
                    val_ct,val_bw,val_labels=val_data["ct"].to(device),val_data["bw"].to(device),val_data["label"].to(device)
                    y_pred=torch.cat([y_pred,model(val_ct,val_bw)],dim=0)
                    y=torch.cat([y,val_labels],dim=0)

                acc_value=torch.eq(y_pred.argmax(dim=1),y)
                acc_metric=acc_value.sum().item()/len(acc_value)
                # y_onehot=[post_label(i) for i in decollate_batch(y)]#?  calculation AUC?
                # y_pred_act=[post_pred(i) for i in decollate_batch(y_pred)]
                # auc_metric(y_pred_act,y_onehot)
                # auc_result=acc_metric.aggregrate()
                # acc_metric.reset()
                # del y_pred_act,y_onehot
                torch.save(model.state_dict(),"C:/YangLi/JHU-PET-CT/data20221013/PSMA_CLASSIFICATION_DATA/Trained model-fuse input-concatenation/classification3d_dict"+str(epoch+1)+".pth")
                if acc_metric>best_metric:
                    best_metric=acc_metric
                    best_metric_epoch=epoch+1
                    torch.save(model.state_dict(),
                               "C:/YangLi/JHU-PET-CT/data20221013/PSMA_CLASSIFICATION_DATA/Trained model-fuse input-concatenation/best_metric_classification3d_dict.pth")
                    print("saved new best metric model")
                print("current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, best_metric, best_metric_epoch))
                writer.add_scalar("val_accuracy",acc_metric,epoch+1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == "__main__":
    main()


