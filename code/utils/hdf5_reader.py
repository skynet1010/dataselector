import torch
import torchvision
import skimage.transform as transform
import numpy as np
import h5py
import os
import cv2

#TODO: optimieren! VIEL Bedarf

class Hdf5Dataset(torch.utils.data.Dataset):
    def __init__(self, in_file_name, training_mode, ds_kind,data_composition, model_key, transform=None):
        super(Hdf5Dataset, self).__init__()
        self.file = h5py.File(in_file_name, "r")
        self.training_mode = training_mode
        self.root_ds_dir = "{}/".format(ds_kind)
        self.dir_dict = {"data":"fus_data","labels":"labels"}
        self.n_images, self.nx, self.ny, self.nz = self.file[self.root_ds_dir+self.dir_dict["data"]].shape
        self.transform = transform
        self.model_key = model_key

        DATA_COMPOSITOR_FUNCTOR = {
            "RGB":self.rgb,
            "NIR":self.nir,
            "SLOPE":self.slope,
            "ROUGHNESS":self.roughness,
            "NDVI":self.ndvi,
            "DOM":self.dom,
            "RGB_NIR":self.rgb_nir,
            "RGB_SLOPE":self.rgb_slope,
            "RGB_NDVI":self.rgb_ndvi,
            "NIR_SLOPE":self.nir_slope,
            "NDVI_SLOPE":self.ndvi_slope,
            "NDVI_NIR":self.ndvi_nir,
            "RGB_NIR_SLOPE":self.rgb_nir_slope,
            "NDVI_NIR_SLOPE":self.ndvi_nir_slope,
            "RGB_NIR_NDVI_SLOPE":self.rgb_nir_ndvi_slope,
        }
        self.data_composition_function = DATA_COMPOSITOR_FUNCTOR[data_composition]

    def get_sample(self,index):
        return cv2.resize(self.file[self.root_ds_dir+self.dir_dict["data"]][index],(224,224) if not self.model_key=="inception" else (299,299)).transpose(2,0,1)

    def rgb(self,sample):
        return sample[:3,:,:]

    def nir(self,sample):
        #return np.resize(sample[3,:,:],(1,self.nx,self.ny))
        return sample[3:4,:,:]

    def slope(self,sample):
        #return np.resize(sample[4,:,:]/90.0,(1,self.nx,self.ny))
        return sample[4:5,:,:]/90.0

    def roughness(self,sample):
        #return np.resize(sample[5,:,:]/78.78074645996094,(1,self.nx,self.ny))
        return sample[5:6,:,:]/78.78074645996094


    def ndvi(self,sample):
        #return np.resize(sample[6,:,:],(1,self.nx,self.ny))
        return sample[6:7,:,:]

    def dom(self,sample):
        #return np.resize(sample[7,:,:]/429.89739990234375,(1,self.nx,self.ny)) #DIESER WERT GILT NUR FÜR DAS ÜBERWACHTE LERNEN!!!!
        return sample[7:8,:,:]/429.89739990234375

    def rgb_nir(self,sample):
        return np.vstack((self.rgb(sample),self.nir(sample)))

    def rgb_slope(self,sample):
        return np.vstack((self.rgb(sample),self.slope(sample)))

    def rgb_ndvi(self,sample):
        return np.vstack((self.rgb(sample),self.ndvi(sample)))

    def nir_slope(self,sample):
        return np.vstack((self.nir(sample),self.slope(sample)))

    def ndvi_slope(self,sample):
        return np.vstack((self.ndvi(sample),self.slope(sample)))

    def ndvi_nir(self,sample):
        return np.vstack((self.ndvi(sample),self.nir(sample)))

    def rgb_nir_slope(self,sample):
        return np.vstack((self.rgb(sample),self.nir(sample),self.slope(sample)))

    def ndvi_nir_slope(self,sample):
        return np.vstack((self.ndvi(sample),self.nir(sample),self.slope(sample)))

    def rgb_nir_ndvi_slope(self,sample):
        return np.vstack((self.rgb(sample),self.nir(sample),self.ndvi(sample),self.slope(sample)))


    def __getitem__(self, index):
        sample = self.get_sample(index)
        data={}
        data["imagery"] = self.data_composition_function(sample)
        nr_of_layers = data["imagery"].shape[0]

        
        if self.training_mode == "supervised":
            data["labels"] = self.file[self.root_ds_dir+self.dir_dict["labels"]][index]
        return data

    def __len__(self):
        return self.n_images

