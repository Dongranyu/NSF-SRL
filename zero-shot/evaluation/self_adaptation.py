import os
import sys

import numpy as np 
import pdb
eps_ = 1e-15

def normalize(x, axis):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)

def cosine_similarity(pred, prototypes):
    nsamples, ndims = np.shape(pred)
    nclasses, _ = np.shape(prototypes)

    pred = pred[:, np.newaxis, :].repeat(nclasses, axis=1)
    prototypes = prototypes[np.newaxis, :, :]

    res = np.sum(normalize(pred, 2) * normalize(prototypes, 2), 2) 
    return res

def eucledian_distance(pred, prototypes):
    nsamples, ndims = np.shape(pred)
    nclasses, _ = np.shape(prototypes)

    pred = pred[:, np.newaxis, :].repeat(nclasses, axis=1)
    prototypes = prototypes[np.newaxis, :, :]
    res = np.sum((pred - prototypes) ** 2, 2)
    return res

def self_adaptation(attr, latent, attr_p_init, step=10, ensemble=True, metric='cosine'):
    ncls, ndims = np.shape(attr_p_init)#10,85# attr_p_init=attribute labels
    def clustering(label):#(7913,)#compute class prototypes
        attr_p_ = np.zeros((ncls, ndims), dtype=np.float32)
        latent_p_ = np.zeros_like(attr_p_)
        count_p = np.zeros((ncls, 1))
        for cls_id in range(ncls):
            
            mask = (label == cls_id)#(7913,)
            attr_p_[cls_id, :] = np.sum(normalize(attr[mask, :], 1), 0)#(85,)
            if metric == 'cosine':
                latent_p_[cls_id, :] = np.sum(normalize(latent[mask, :], 1), 0) # partial best solution#[10,85]
            elif metric == 'euclidean':
                latent_p_[cls_id, :] = np.sum(latent[mask, :], 0) # partial best solution
            count_p[cls_id, :] = np.sum(mask)
        attr_p_ = attr_p_ / (count_p + eps_)#[10,85]
        latent_p_ = latent_p_ / (count_p + eps_)#[10,85]
        
        if 0 in count_p:
            cls_mask = (count_p == 0)
            mean_attr = np.sum(attr_p_ * count_p, 0) / np.sum(count_p)
            mean_latent = np.sum(latent_p_ * count_p, 0) / np.sum(count_p)
            attr_p_[cls_mask[:, 0], :] = mean_attr
            latent_p_[cls_mask[:, 0], :] = mean_latent
        return attr_p_, latent_p_
    
    def labeling(attr_p_, latent_p_=None):
        
        #attr=[7913,85],attr_p_=[10,85]
        
        cos_sim_attr = cosine_similarity(attr, attr_p_)#[7913,10]
        if latent_p_ is not None:
            if metric == 'cosine':
                latent_sim = cosine_similarity(latent, latent_p_)
            elif metric == 'euclidean':
                latent_sim = 1 / (1 + eucledian_distance(latent, latent_p_))
                cos_sim_attr = normalize(cos_sim_attr, 1)
                latent_sim = normalize(latent_sim, 1)
        else:
            latent_sim = np.zeros_like(cos_sim_attr)
        
        if ensemble or latent_p_ is not None:
            final_sim = cos_sim_attr + latent_sim
        else:
            final_sim = cos_sim_attr
        
        return final_sim.argmax(1)#prediction class label
    
    attr_p, latent_p = attr_p_init, None #attr_p = attribute label
    
    labels = []
    for i in range(step):
        pseudo_label = labeling(attr_p, latent_p)#(7913,)array([3, 0, 7, ..., 1, 4, 1])
        attr_p, latent_p = clustering(pseudo_label)
        labels.append(pseudo_label)
   
    return labels#10[array([3, 0, 7, ..., 1, 4, 1]), array([6, 0, 1, ..., 1, 9, 3])]

