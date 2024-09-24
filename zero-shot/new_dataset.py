from os.path import join as joinpath
#from preprocess import data_process
from preprocess import preprocess
from common.constants import const_dict
from common.predicate import PRED_DICT
import copy
import random
import numpy as np
from random import shuffle,choice
import json
import itertools
import pdb

class Dataset():

    def __init__(self,data_root):
       
        fact_ls, rule_ls = preprocess(joinpath(data_root, 'relations.txt'),
                                                        joinpath(data_root, 'facts.txt'),
                                                        joinpath(data_root, 'rules.txt')
                                                        )
        # fact_ls,rule_ls,pred_id2name = data_process(joinpath(data_root,"relations_SS.txt"), \
        #                                             joinpath(data_root,"facts_SS.txt"), \
        #                                             joinpath(data_root,"rules_SS.txt"))
      
        self.rule_ls = rule_ls
        #self.fact_dict = self.cls_pred_total(data_root)
        self.fact_ls = fact_ls
        #self.mln_fact = self.rand_del_fact()
        self.pred_id2name =self.generate_id2rel('/data/ydr2021/image_classification_CUB/AttentionZSL/data/CUB/mln/predicates.json')
        self.num_rules = len(rule_ls)

    def generate_id2rel(self,path):
        rel = json.load(open(path))
        rel_dict = dict((rel.index(name),name) for name in rel)
        return rel_dict
    
    def rand_del_fact(self):

        del_key_id=random.sample(range(0,len(self.fact_ls)),2000)
        
        new_fact = copy.deepcopy(self.fact_ls)
       
        for k in del_key_id:
            del new_fact[k]
        return new_fact


    def cls_id2name(self,dataroot):
        box_obj = dict()
        with open(joinpath(dataroot,"ent2class_PS.txt")) as f:
            for line in f:
                box,obj_id = line.strip().split(' ')
                box_obj[box] = int(obj_id)
        
        return box_obj


    def find_rule(self, sub):#sub是[index, class_id]之类的索引
        samples_by_r = []
        neg_mask_by_r = []
        obs_mask_by_r = []
        latent_mask_by_r = []
        obs_var_by_r = []
  
        r_index = []
        n=0
       
        for instance in sub:
           
            x = instance[1].item()#class_id
            instance_name = self.pred_id2name[x]
            
            for r_id,formula in enumerate(self.rule_ls):
                
               
                if instance_name==formula.atom_ls[-1].pred_name:
                    
                    ins = {'x':x}
                    samples,latent_mask,obs_mask,obs_var,neg_mask = self.get_rule_rnd(ins,formula)
                    samples_by_r.append(samples)
                    latent_mask_by_r.append(latent_mask)
                    obs_mask_by_r.append(obs_mask)
                    obs_var_by_r.append(obs_var)
                    neg_mask_by_r.append(neg_mask)
                    r_index.append(r_id)
                    n=n+1
                    break    
                else:
                    continue
        
        return samples_by_r,latent_mask_by_r,obs_mask_by_r, obs_var_by_r,neg_mask_by_r

    def get_rule_rnd(self,ins,rule):
        
        samples = [[atom.pred_name, []] for atom in rule.atom_ls]  # [规则对应的原子的谓词,[]] 把那条规则对应的原子集取出来
        
        latent_mask = [[atom.pred_name, []] for atom in rule.atom_ls] #记录隐变量
        obs_mask = [[atom.pred_name, []] for atom in rule.atom_ls] ##记录观测变量
        obs_var = [[atom.pred_name, []] for atom in rule.atom_ls]
        neg_mask = [[atom.pred_name, []] for atom in rule.atom_ls]
        # for index,atom in enumerate(rule.atom_ls):
        #     if index < len(rule.atom_ls)-1:
        #         var = ins[atom.var_name_ls[0]]
        #         samples[index][1].append(int(var))
        #         latent_mask[index][1].append(1) # 1 represents latent variable
        #     else:
    
        #         var = ins[atom.var_name_ls[0]]
        #         samples[index][1].append(int(var))
        #         latent_mask[index][1].append(0) # 0 represents observed variable  
        for index,atom in enumerate(rule.atom_ls):
            if index < len(rule.atom_ls)-1:
                var = ins[atom.var_name_ls[0]]
                samples[index][1].append(int(var))
                obs_var[index][1].append(int(var))
                #val =  np.random.choice([0,1])# 1 represents latent variable#规则体随机选择隐变量
                val =1 #All atoms in rule body are latent variable 
                latent_mask[index][1].append(val) # 1 represents latent variable#规则体随机选择隐变量
            else:
    
                var = ins[atom.var_name_ls[0]]
                samples[index][1].append(int(var))
                obs_var[index][1].append(int(var))
                #latent_mask[index][1].append(0) # 0 represents observed variable #规则头为观测变量 
                latent_mask[index][1].append(1)
            if atom.neg == True:
               
                neg_mask[index][1].append(0)
            else:
                neg_mask[index][1].append(1)
        return samples,latent_mask,obs_mask, obs_var, neg_mask

   
    
######################test
class test_Dataset():

    def __init__(self,data_root):

        
        # fact_ls, rule_ls, query_ls = preprocess(joinpath(data_root, 'relations.txt'),
        #                                                 joinpath(data_root, 'facts.txt'),
        #                                                 joinpath(data_root, 'rules.txt'),
        #                                                 joinpath(data_root, 'queries.txt'))
        fact_ls, rule_ls = preprocess(joinpath(data_root, 'relations.txt'),
                                                        joinpath(data_root, 'facts.txt'),
                                                        joinpath(data_root, 'rules.txt'))
        self.rule_ls = rule_ls
        #self.fact_dict = self.cls_pred_total(data_root)
        self.fact_ls = fact_ls
        #self.mln_fact = self.rand_del_fact()
        self.pred_id2name =self.generate_id2rel('/data/ydr2021/image_classification_CUB/AttentionZSL/data/CUB/mln/predicates.json')
        self.num_rules = len(rule_ls)

    def generate_id2rel(self,path):
        rel = json.load(open(path))
        rel_dict = dict((rel.index(name),name) for name in rel)
        return rel_dict
    # def rand_del_fact(self):

    #     del_key_id=random.sample(range(0,len(self.fact_ls)),2000)
        
    #     new_fact = copy.deepcopy(self.fact_ls)
       
    #     for k in del_key_id:
    #         del new_fact[k]
    #     return new_fact

    
    # def cls_id2name(self,dataroot):
    #     box_obj = dict()
    #     with open(joinpath(dataroot,"ent2class_SS_test.txt")) as f:
    #         for line in f:
    #             box,obj_id = line.strip().split(' ')
    #             box_obj[box] = int(obj_id)
        
    #     return box_obj

    def generate_rel2id(self,path):
        rel = json.load(open(path))
        rel_dict = dict((name,rel.index(name)) for name in rel)
        return  rel_dict
    
    def find_rule(self, sub,class_name):#sub是[index, class_id]之类的索引
        
        samples_by_r = []
       
        for instance in sub:
            
            x = instance#class_id
            instance_name = class_name[x]
           
      
            for r_id,formula in enumerate(self.rule_ls):
               
                if instance_name==formula.atom_ls[-1].pred_name:
                    
                    ins = {'x':x}
                    samples= self.get_rule_rnd(ins,formula)
                    samples_by_r.append(samples)
            
                    break    
                else:
                    continue
        
        return samples_by_r

    def get_rule_rnd(self,ins,rule):
        
        samples = [[atom.pred_name, []] for atom in rule.atom_ls]  # [规则对应的原子的谓词,[]] 把那条规则对应的原子集取出来
        
        
        # for index,atom in enumerate(rule.atom_ls):
        #     if index < len(rule.atom_ls)-1:
        #         var = ins[atom.var_name_ls[0]]
        #         samples[index][1].append(int(var))
        #         latent_mask[index][1].append(1) # 1 represents latent variable
        #     else:
    
        #         var = ins[atom.var_name_ls[0]]
        #         samples[index][1].append(int(var))
        #         latent_mask[index][1].append(0) # 0 represents observed variable  
        for index,atom in enumerate(rule.atom_ls):
            if index < len(rule.atom_ls)-1:
                var = ins[atom.var_name_ls[0]]
                samples[index][1].append(int(var))
                
            else:
    
                var = ins[atom.var_name_ls[0]]
                samples[index][1].append(int(var))
               
        return samples