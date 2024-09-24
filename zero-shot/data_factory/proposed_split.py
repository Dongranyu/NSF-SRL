import os
from os.path import join 
import numpy as np
import torch
from torch.utils.data import Dataset
from random import shuffle
from copy import deepcopy
from util import *
from .data_transform import data_transform
import pdb
class PSFactory(Dataset):
    def __init__(self, 
                data_path, 
                attr_file, 
                cls_file, 
                train_file, 
                transform,
                batch_size,
                im_size,
                triplet_selections=4):
        self.triplet_k = triplet_selections
        self.triplet_p = batch_size // self.triplet_k

        self.transform = data_transform(transform, im_size[0])
        self.dataset_path = data_path
        

        if not os.path.exists(self.dataset_path):
            raise RuntimeError('[!] dataset not found: {}'.format(self.dataset_path))
        
        self.all_attr = prepare_attribute_matrix(attr_file)#[50,85]
        all_cls_names = prepare_cls_names(cls_file)#[]#50
        
        self.all_train_files = loadtxt(train_file)#23527
        all_train_cls = [f[:f.find('/')] for f in self.all_train_files]##23527
      
        self.factory_cls_names = sorted(list(set(all_train_cls)))#40
        

        cls_indice = []#category index
        for cls in self.factory_cls_names:
            if cls in all_cls_names:
                cls_indice.append(all_cls_names.index(cls))
        
        assert(len(cls_indice) == len(self.factory_cls_names))
        self.attr_selected = self.all_attr[np.asarray(cls_indice), :]#[150,312]
      
        self.all_im_names = [os.path.join(self.dataset_path, im_path) for im_path in self.all_train_files]#23527
        # Build File Dictionary
        self.file_dict = {}
        self.batch_count = {}
        self.batch_sentry = {}
        self._length = 0
        self._size   = 0
        for cls, full_path in zip(all_train_cls, self.all_im_names):
            if cls in self.file_dict:
                self.file_dict[cls].append(full_path)
            else:
                self.file_dict[cls] = [full_path]

        for cls in self.factory_cls_names:
            shuffle(self.file_dict[cls])
            self.batch_count[cls] = len(self.file_dict[cls]) // self.triplet_k
            self.batch_sentry[cls] = 0
            self._length += self.batch_count[cls]
            self._size += len(self.file_dict[cls])
       
        self.factory_cls_names_cp = deepcopy(self.factory_cls_names)

    def __len__(self):
        return self._length // self.triplet_p

    def __getitem__(self, index):
        batch_cls = []
        im, attr_indice = [], []
        for i in range(self.triplet_p):
            selected_cls = randpick(self.factory_cls_names_cp, exception=batch_cls)
            batch_cls.append(selected_cls)
            sentry = self.batch_sentry[selected_cls]
            for j in range(self.triplet_k):
                
                im.append(self.transform(load_pilimage(
                    self.file_dict[selected_cls][sentry * self.triplet_k + j])))
                
                # img = '/data/ydr2021/image_classification_CUB/AttentionZSL/data/CUB/JPEGImages/001.Black_footed_Albatross/Black_Footed_Albatross_0005_796090.jpg'
                # test_feature = self.transform(load_pilimage(img))
                # torch.save(test_feature,'/data/ydr2021/image_classification_CUB/AttentionZSL/result_images/visual_data/heat_images/Black_Footed_Albatross_0005_796090.json')
             
                attr_indice.append(self.factory_cls_names.index(selected_cls))
            self.batch_sentry[selected_cls] += 1
            if self.batch_sentry[selected_cls] == self.batch_count[selected_cls]:
                self.factory_cls_names_cp.remove(selected_cls)

        if len(self.factory_cls_names_cp) < self.triplet_p:
            self.factory_cls_names_cp = deepcopy(self.factory_cls_names)
            for cls in self.factory_cls_names:
                shuffle(self.file_dict[cls])
                self.batch_sentry[cls] = 0
       
        im = torch.cat(im, dim=0)
        attr_mask = np.zeros((len(attr_indice), len(self.factory_cls_names)), dtype=np.float32)#[24,40]
        
        attr_label = self.all_attr[np.asarray(attr_indice), :]#[24,40]
       
        for i, j in enumerate(attr_indice):
            attr_mask[i, j] = 1
      
        return im, attr_mask, attr_label

    def selected_attr(self):
        return self.attr_selected

    def size(self):
        return self._size
 
