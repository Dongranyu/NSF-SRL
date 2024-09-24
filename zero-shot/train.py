import os
import sys
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim import Adagrad
from torch.utils.data import DataLoader
from itertools import chain
import models
import data_factory
import losses
from glob import glob
import pdb
from mean_field_posterior import FactorizedPosterior
from new_dataset import Dataset 
def compute_entropy(posterior_prob):
  eps = 1e-6
  posterior_prob.clamp_(eps, 1 - eps)
  compl_prob = 1 - posterior_prob
  entropy = -(posterior_prob * torch.log(posterior_prob) + compl_prob * torch.log(compl_prob)).sum()
  return entropy
class trainer(object):
    def __init__(self, args, cfg, checkpoint_dir):
        self.batch_size = cfg.train.batch_size
        self.learning_rate = cfg.train.lr
        self.epochs = cfg.train.epochs
        self.start_epoch = 1
        self.lr_decay_epochs = cfg.train.lr_decay
        self.log_interval = cfg.train.log_inter
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = cfg.train.ckpt_inter
        self.lambda_ = cfg.train.beta
        self.attr_dims = cfg.attr_dims
        self.device = torch.device(
            'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        self.triplet_batch = 4
        
        self.rule_dataset = Dataset('/data/ydr2021/image_classification_CUB/AttentionZSL/data/CUB/mln')
        torch.cuda.set_device(0)
        # self.posterior =  FactorizedPosterior(self.attr_dims, 5).to(self.device)
        self.posterior =  FactorizedPosterior(self.attr_dims, 5)
        self.posterior = torch.nn.DataParallel(self.posterior, device_ids=[0,1,2,3])

        self.fnet, self.optimizer, self.im_size,self.all_params = self.build_model(cfg)
        if os.path.exists(cfg.ckpt_name) and args.fine_tuning:
            pth = glob(os.path.join(cfg.ckpt_name, "ckpt_epoch_*.pth"))
            pth = sorted(pth, 
                         key=lambda p: int(os.path.basename(p).replace("ckpt_epoch_", "").replace(".pth", "")), 
                         reverse=True)
            if pth:
                self.load(pth[0])
                self.start_epoch = int(
                        ''.join([c for c in os.path.basename(pth[0]) if c.isdigit()])
                        ) + 1
            
        self.attr_data, self.dataset_size, self.data_loader = self.prepare_dataloader(cfg)
        #self.attr_data = torch.from_numpy(self.attr_data).to(self.device)
        self.online_zsl_loss = losses.ZeroShotLearningLoss(self.attr_data)
       
        if cfg.train.triplet_mode == "batch_all":
            self.online_triplet_loss = \
                        losses.BatchAllTripletLoss(self.device, 
                                                   self.batch_size // self.triplet_batch, 
                                                   self.triplet_batch)
        else:
            self.online_triplet_loss = \
                        losses.BatchHardTripletLoss(self.device,
                                                    self.batch_size // self.triplet_batch,
                                                    self.triplet_batch)

    def build_model(self, cfg):
        fnet, im_size = models.load_model(cfg.model, k=self.attr_dims)
        all_params = chain.from_iterable([fnet.parameters(), self.posterior.parameters()])
        optimizer = Adam(all_params, self.learning_rate)
        fnet=torch.nn.DataParallel(fnet, device_ids=[0,1,2,3])
        return fnet, optimizer, im_size,all_params

    def prepare_dataloader(self, cfg):
        if cfg.split == "SS":
            dataset = data_factory.SSFactory(
                cfg.image, cfg.attribute, cfg.class_name, cfg.ss_train, 
                transform=cfg.train.data_aug, batch_size=self.batch_size, im_size=self.im_size
            )
        elif cfg.split == "PS":
            dataset = data_factory.PSFactory(
                cfg.image, cfg.attribute, cfg.class_name, cfg.ps_train, 
                transform=cfg.train.data_aug, batch_size=self.batch_size, im_size=self.im_size
            )
        else:
            raise NotImplementedError
        attr_data = dataset.selected_attr()
        attr_data = torch.from_numpy(attr_data).to(self.device)
        dataset_size = dataset.size()

        dataset.im_size = self.im_size
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        return attr_data, dataset_size, data_loader

    def exp_lr_scheduler(self, epoch, lr_decay_epoch, lr_decay=0.1):
        if epoch % lr_decay_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay

    def run(self):
        for e in range(self.start_epoch, self.epochs + self.start_epoch):
            self.exp_lr_scheduler(e, self.lr_decay_epochs)
            self.fnet.train()
            self.posterior.train()
            agg_loss = {"loss": 0., "attr_loss": 0., "latent_loss": 0, "mln_loss": 0,}
            current_size = 0
            for batch_id, (x, attr_mask, attr_label) in enumerate(self.data_loader):
                current_size += self.batch_size
              
                x = x.to(self.device)   # 1 x 3#batch_size=24 x 224 x 224#[72,224 x 224]
                attr_mask = attr_mask.to(self.device).squeeze() # 1 x #batch_size x k#[batch,40]
                attr_label = attr_label.to(self.device).squeeze()#[batch,85]
                
                attr_embed, latent_embed = \
                        self.fnet(x.view(-1, 3, x.size(2), x.size(3)))
                constant_embedding = attr_embed
                #latent_embed = attr_embed#yu add

                latent_embed = latent_embed.view(
                        self.batch_size // self.triplet_batch, self.triplet_batch, latent_embed.size(1))
                
                latent_loss = self.online_triplet_loss(latent_embed)
                attr_loss = self.online_zsl_loss(attr_embed,
                                                 attr_mask)
                
                #####################################################################add MLN
                rule_dataset = Dataset('/data/ydr2021/image_classification_CUB/AttentionZSL/data/CUB/mln')
                
                sub = torch.nonzero(attr_mask==1).squeeze()#return index of the no-zero value #find class id
                
        
                samples_by_r,latent_mask_by_r,obs_mask_by_r, obs_var_by_r,neg_mask_by_r = rule_dataset.find_rule(sub)
                mln_loss = 0.0
               
                for ind, samples in enumerate(samples_by_r):
                  
                    latent_mask = latent_mask_by_r[ind]
                    obs_mask = obs_mask_by_r[ind]
                    obs_var = obs_var_by_r[ind]
                    neg_mask = neg_mask_by_r[ind]
                    one_constant_embedding = constant_embedding[ind]
                    
                 
                  

                    potential, posterior_prob, obs_xent, reasoning_attr = self.posterior([samples, latent_mask, obs_mask,obs_var, neg_mask], one_constant_embedding,training = True)
                    entropy = compute_entropy(posterior_prob)
                    
                    mln_loss += - (potential.sum() * 1 + entropy) / (potential.size(0) + 1e-6) + obs_xent
                    
                
                mln_loss = mln_loss / len(samples_by_r)
            
               
               

                #attr_loss = attr_loss * self.lambda_

                #loss = latent_loss + attr_loss + mln_loss + link_loss + attr_regression_loss
                #loss = latent_loss + attr_loss + mln_loss + attr_regression_loss
                #loss = latent_loss + attr_loss + mln_loss 
                loss = latent_loss + attr_loss + 0.1*mln_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fnet.parameters(), 5)
                #torch.nn.utils.clip_grad_norm_(self.posterior.parameters(), 5)
                self.optimizer.step()

                agg_loss["loss"] += loss.item()
                agg_loss["attr_loss"] += attr_loss.item()
                agg_loss["latent_loss"] += latent_loss.item()
                agg_loss["mln_loss"] += mln_loss.item()
                if current_size % self.log_interval == 0:
                    mesg = "[E{} {}/{} Cur/Agg]\t latent_loss:{:.3f}/{:.3f}\t attr_loss:{:.3f}/{:.3f}\t mln_loss:{:.3f}/{:.3f}\t total:{:.3f}/{:.3f}".format(
                        e, current_size, self.dataset_size,
                        latent_loss.item(),
                        agg_loss["latent_loss"] / (batch_id + 1), 
                        attr_loss.item(),
                        agg_loss["attr_loss"] / (batch_id + 1),
                        mln_loss.item(),
                        agg_loss["mln_loss"] / (batch_id + 1),
                        loss.item(),
                        agg_loss["loss"] / (batch_id + 1)
                    )
                    print(mesg)
            
            if self.checkpoint_dir is not None and e % self.checkpoint_interval == 0:
                self.save(self.checkpoint_dir, e)

    def save(self, checkpoint_dir, e):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.fnet.eval()
        ckpt_model_filename = os.path.join(checkpoint_dir, "ckpt_epoch_" + str(e) + ".pth")
        state_dict = self.fnet.state_dict()
        torch.save(state_dict, ckpt_model_filename)
        
        self.posterior.eval()
        concept_ckpt_model_filename = os.path.join(checkpoint_dir, "concept_network_ckpt_epoch_" + str(e) + ".pth")
        torch.save(self.posterior.state_dict(),concept_ckpt_model_filename)

        self.fnet.train()
        #self.posterior.train()
    def load(self, checkpoint_dir):
        state_dict = torch.load(checkpoint_dir)
        
        self.fnet.load_state_dict(state_dict)
        self.fnet.to(self.device)