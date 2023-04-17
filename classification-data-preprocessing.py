# spacing    in training & testing
# crop
import nibabel as nb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk

data = pd.read_excel('C:/YangLi/JHU-PET-CT/data20221013/PSMA_CLASSIFICATION_DATA/ROIs.xlsx',sheet_name='ROIs_mask_label')
for i in range(0,3214):
   print(i,data.iloc[i, 0],data.iloc[i, 2],)
   #roi=np.zeros([481,363,113])
   ctpath=r'C:\YangLi\JHU-PET-CT\data20221013\PSMACADX_FINAL_RT_nii'
   suvbwpath=r'C:\YangLi\JHU-PET-CT\data20221013\PSMA-SUV'
   ctname=ctpath+'/'+data.iloc[i, 0]+'/CT/image.nii.gz'
   ctlabel =ctpath+'/'+data.iloc[i, 0]+'/CT/mask_'+data.iloc[i, 2]+'.nii.gz'

   suv=suvbwpath+'/'+data.iloc[i, 0]
   patientdir = os.listdir(suv)
   suv1=suv+'/'+patientdir[2]
   patientdir1 = os.listdir(suv1)
   suv2 = suv1 + '/' + patientdir1[0]
   suvbwname=suv2+'/'+os.listdir(suv2)[2]
   suvbwlabel=ctpath+'/'+data.iloc[i, 0]+'/PET/mask'+data.iloc[i, 2]+'.nii.gz'

   bw=sitk.Image(sitk.ReadImage(suvbwname))
   sitk.WriteImage(bw, suv2+'/' +'SUVbw.nii.gz')

   ctimage = nb.load(suv2+'/' +'SUVbw.nii.gz')
   ct_imdata=ctimage.get_fdata()

   ctlabel_data = nb.load(suvbwlabel)
   ct_labdata = ctlabel_data.get_fdata()
   affinect=ctimage.affine.copy()
   hdrct=ctimage.header.copy()

   x,y,z=np.nonzero(ct_labdata)

   ct_roi=ct_imdata[np.min(x):np.max(x)+1,np.min(y):np.max(y)+1,np.min(z):np.max(z)+1] #roi


   # plt.imshow(roi[:,:,0])
   # plt.show()
   # plt.imshow(roi[:, :,56])
   # plt.show()

   file='C:/YangLi/JHU-PET-CT/data20221013/PSMA_CLASSIFICATION_DATA/SUVbw/'+data.iloc[i,0]+data.iloc[i,2]+'.nii.gz'

   new_roi=nb.Nifti1Image(ct_roi,affinect,hdrct)
   nb.save(new_roi,file)
   # plt.imshow(roi[:,:,0])
   # plt.show()
   del x,y,z,new_roi

print("finish")