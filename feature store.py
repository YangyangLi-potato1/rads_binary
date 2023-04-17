import logging
import os
import sys
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import monai
from monai.data import CSVSaver
from monai.transforms import AddChanneld, Compose, LoadImaged, Resized, ScaleIntensityd, EnsureTyped,Spacingd,RandRotate90d
from networks.densenet import DenseNet
# from networks.densenet_fuse_input import DenseNet  #fuse the input with 3D conv

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    ct_path = r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/CT/'
    bw_path = r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/SUVbw/'
    xx = pd.read_excel(r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/ROIs.xlsx',
                       sheet_name='ROIs_mask_label')
    images_ct = xx.iloc[:, 4].tolist()  # all ct images
    images_bw = xx.iloc[:, 5].tolist()  # all suvbw images
    lab = xx.iloc[:, 3].tolist()  # all labels

    labels_all = np.array(lab, dtype=np.int64)

    ct = images_ct[0:3209]  # train+val images
    bw = images_bw[0:3209]
    labels = labels_all[0:3209]  # train+val labels

    ct1 = [os.sep.join([ct_path, f]) for f in ct]
    bw1 = [os.sep.join([bw_path, f]) for f in bw]

    train_files = [{"ct": ct2 + '.nii.gz', "bw": bw2 + '.nii.gz', "label": label} for ct2, bw2, label in
                   zip(ct1[:2258], bw1[:2258], labels[:2258])]  # train images,labels
    # val_files = [{"ct": ct2 + '.nii.gz', "bw": bw2 + '.nii.gz', "label": label} for ct2, bw2, label in
    #              zip(ct1[2258:], bw1[2258:], labels[2258:])]  # val images,labels
    val_files = [{"ct": ct2 + '.nii.gz', "bw": bw2 + '.nii.gz', "label": label} for ct2, bw2, label in
                 zip(ct1[:], bw1[:], labels[:])]  # val images,labels

    #
    val_transforms = Compose([LoadImaged(keys=["ct", "bw"]),
                              AddChanneld(keys=["ct", "bw"]),
                              ScaleIntensityd(keys=["ct", "bw"]),
                              Spacingd(keys=["ct", "bw"], pixdim=(1, 1, 1), mode=("bilinear")),
                              Resized(keys=["ct", "bw"], spatial_size=(48, 48, 48)),
                              RandRotate90d(keys=["ct", "bw"], prob=0.8, spatial_axes=[0, 2]),
                              EnsureTyped(keys=["ct", "bw"])])

    #
    val_ds=monai.data.Dataset(data=val_files,transform=val_transforms)
    val_loader=DataLoader(val_ds,batch_size=1,num_workers=4,pin_memory=torch.cuda.is_available())

    #
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model=monai.networks.nets.DenseNet121(spatial_dims=3,in_channels=1,out_channels=2).to(device)
    model = DenseNet(spatial_dims=3, in_channels=1, out_channels=2, init_features=64,
                     growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, act=('relu', {'inplace': True}),
                     norm='batch', dropout_prob=0.0).to(device)

    model.load_state_dict(torch.load("C:/YangLi/JHU-PET-CT/data20221013/PSMA_CLASSIFICATION_DATA/Trained model-bw/best_metric_classification3d_dict.pth"))
    # model.load_state_dict(torch.load("C:/YangLi/JHU-PET-CT/data20221013/PSMA_CLASSIFICATION_DATA/Trained model-ct/classification3d_dict850.pth"))
    model.eval()

    with torch.no_grad():
        num_correct=0
        metric_count=0
        saver=CSVSaver(output_dir="./output-CT-single-feature")
        for val_data in val_loader:
            val_ct,val_bw,val_labels=val_data["ct"].to(device),val_data["bw"].to(device),val_data["label"].to(device)
            # val_outputs=model(val_images,val_cli).argmax(dim=1)
            val_outputs,val_feature = model(val_bw)
            val_outputs1 = val_outputs.argmax(dim=1)
            val_outputs2=F.softmax(val_outputs,dim=1)  #final column is the probability
            # val_outputs=val_outputs.argmax(dim=1)
            print(val_outputs1, val_labels)
            value=torch.eq(val_outputs,val_labels)
            metric_count+=len(value)
            num_correct+=value.sum().item()
            # saver.save_batch(val_outputs2)
            # saver.save_batch(val_outputs1)
            saver.save_batch(val_feature)
            #saver.save_batch(val_outputs,val_data["img_meta_dict"])
        metric=num_correct/metric_count
        print("evalution metric:",metric)
        saver.finalize()

if __name__=="__main__":
    main()


