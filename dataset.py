import os 
import sys
import random
import math
import numpy as np

from tqdm import tqdm
import glob
import pickle


import warnings 
warnings.filterwarnings("ignore")


segmentation_train_dir="/home/ye/Data/Image/Indian-Diabetic/Segmentation/Groundtruths/training/"
original_train_dir="/home/ye/Data/Image/Indian-Diabetic/Segmentation/Original-Images/training/"

segmentation_val_dir="/home/ye/Data/Image/Indian-Diabetic/Segmentation/Groundtruths/testing/"
original_val_dir="/home/ye/Data/Image/Indian-Diabetic/Segmentation/Original-Images/testing//"
#images_list=glob.glob(DATA_DIR+"*.tif")


height,width=2848,4288


# function for create dataset
def get_tif_fps(tif_dir,pathern="*.tif"):
    tif_fps=glob.glob(tif_dir+pathern)
    return list(set(tif_fps))

def get_tif_imageid(tif_fps,pathern="_"):
    tif_fps=os.path.basename(tif_fps)
    tif_part=tif_fps.split(pathern)
    image_id=tif_part[0]+"_"+tif_part[1]
    class_id=tif_part[2].split(".")[0]
    return image_id,class_id

def segmentation_to_original(image_id,original_dir):
    img_path=image_id+".jpg"
    img_path=original_dir+img_path
    #imgs=get_tif_fps(original_dir,".jpg")
    return img_path


def class_mask(segmentation_dir,pathern="*.tif"):
    classes=os.listdir(segmentation_dir)
    image_mask={fps:[] for fps in classes}
    image_id={fps:[] for fps in classes}
    image_path={fps:[] for fps in classes}
    for ann in tqdm(classes):
        path=os.path.join(segmentation_dir,ann)
        fps=get_tif_fps(path,pathern)
        for seg in fps:
            mask=cv2.imread(seg,0)
            Id,_=get_tif_imageid(seg,"_")
            #mask=np.reshape(mask,[mask.shape[0],mask.shape[1],1])
            image_mask[ann].append(mask)
            image_id[ann].append(Id)
            image_path[ann].append(seg)
    return image_mask,image_id,image_path


def annotation(IDs,ID,mask_pathes,original_dir):
    
    ann=[]
    mask_path=[]
    for key,value in IDs.items():
        if ID in value:
            index=IDs[key].index(ID)
            mask_path.append(mask_pathes[key][index])
            path=os.path.join(original_dir,ID)+".jpg"
            ann.append(key)
    return {"classes":ann,"id":ID,"path":path,"mask_path":mask_path}
            

def merge_mask(idx,IDs,mask_pathes,original_dir,height,width):
    """
    args:
      idx: the image id
      IDs: all images id sets( a dict)
      mask_pathes: all the mask tif file pathes of each classes(a dict)
      original_dir:original image path
      height: mask height
      width: mask width
    
    return: the idx image annotation
    """
    #classes=list(maskes.keys())
    ann=annotation(IDs,idx,mask_pathes,original_dir)
    mask_zero=np.zeros((height,width,len(ann["mask_path"])))
    for i,msk in enumerate(ann["mask_path"]):
        x=cv2.imread(msk,0)
        assert (height,width)==x.shape
        mask_zero[:,:,i]=x
    ann["mask"]=mask_zero.astype(bool)
    #Anns.append(ann)
    return ann


# make dataset

#val_maskes,val_IDs,val_mask_pathes=class_mask(segmentation_val_dir,pathern="/*.tif")
#height,width=2848,4288

#id_set=[]
#for k,v in val_IDs.items():
#    id_set.extend(v)
#id_set=np.unique(id_set)

#val_mask=list(map(lambda x:merge_mask(idx=x,IDs=val_IDs,mask_pathes=val_mask_pathes,original_dir=original_val_dir,height=height,width=width),id_set))
#with open("./data/val-mask.pkl","wb") as msk:
#    pickle.dump(val_mask,msk)
