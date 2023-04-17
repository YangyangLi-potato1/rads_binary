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
from monai.transforms import AddChanneld, Compose, LoadImaged, Resized, ScaleIntensityd, EnsureTyped
from networks.feature_fusion import FF

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    ct = pd.read_csv(r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/ct-feature.csv')
    ct_feature_all = np.array(ct)  # [533,1041]
    ct_feature = ct_feature_all[:, 1:]
    ct_feature = np.array(ct_feature, dtype=np.float)

    bw = pd.read_csv(r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/bw-feature.csv')
    bw_feature_all = np.array(bw)  # [3209,1025]
    bw_feature = bw_feature_all[:, 1:]
    bw_feature = np.array(bw_feature, dtype=np.float)

    xx = pd.read_excel(r'C:\YangLi\JHU-PET-CT\data20221013\PSMA_CLASSIFICATION_DATA/ROIs.xlsx',
                       sheet_name='ROIs_mask_label')
    labels = xx.iloc[:, 3].tolist()  # all labels


    val_files = [{"label": label, "ct": ct, "bw": bw} for label, ct, bw in
                 zip(labels[2258:], ct_feature[2258:, :], bw_feature[2258:, :])]


    #
    val_ds=monai.data.Dataset(data=val_files)
    val_loader=DataLoader(val_ds,batch_size=1,num_workers=4,pin_memory=torch.cuda.is_available())

    #
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model=monai.networks.nets.DenseNet121(spatial_dims=3,in_channels=1,out_channels=2).to(device)
    model = FF().to(device)

    model.load_state_dict(torch.load("C:/YangLi/JHU-PET-CT/data20221013/PSMA_CLASSIFICATION_DATA/Trained model-fuse output-feature Transformer/best_metric_classification3d_dict.pth"))
    model.eval()

    with torch.no_grad():
        num_correct=0
        metric_count=0
        saver=CSVSaver(output_dir="./output-fuse feature Transformer01")
        for val_data in val_loader:
            val_ct, val_bw, val_labels = val_data["ct"].to(device), val_data["bw"].to(device), val_data["label"].to(
                device)
            # val_outputs=model(val_images,val_cli).argmax(dim=1)
            val_outputs = model(val_ct,val_bw)
            val_outputs1 = val_outputs.argmax(dim=1)
            val_outputs2 = F.softmax(val_outputs, dim=1)  # final column is the probability
            print(val_outputs1, val_labels)
            value=torch.eq(val_outputs,val_labels)
            metric_count+=len(value)
            num_correct+=value.sum().item()
            saver.save_batch(val_outputs1)
        metric=num_correct/metric_count
        print("evalution metric:",metric)
        saver.finalize()

if __name__=="__main__":
    main()


