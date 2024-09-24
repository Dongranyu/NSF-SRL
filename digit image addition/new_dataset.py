from os.path import join as joinpath
from preprocess import data_process
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

        rpath = joinpath(data_root,"rules.txt")
        fact_ls,rule_ls,pred_id2name = data_process(joinpath(data_root,"relations.txt"), \
                                                    joinpath(data_root,"fact.txt"), \
                                                    joinpath(data_root,"entity_class.txt"), \
                                                    rpath)
      
        self.rule_ls = rule_ls
        self.fact_dict = self.cls_pred_total(data_root)
        self.mln_fact = self.rand_del_fact()
        self.pred_id2name = pred_id2name
        self.num_rules = len(rule_ls)

    def rand_del_fact(self):

        del_key_id=random.sample(range(0,len(self.fact_dict)),20000)
        # print(max(del_key_id))
        new_fact = copy.deepcopy(self.fact_dict)
        # print(len(new_fact))
        for k in del_key_id:
            # print(k)
            key = list(self.fact_dict.keys())[k]
            del new_fact[key]
        return new_fact


    def cls_id2name(self,dataroot):
        box_obj = dict()
        with open(joinpath(dataroot,"entity_class.txt")) as f:
            for line in f:
                box,obj_id = line.strip().split(' ')
                box_obj[box] = int(obj_id)-19
        # box_name = dict()
        # obj = json.load(dataroot,"objects.json")
        # for k in box_obj.keys():
        #     box_name[k] = obj[box_obj[k]]

        return box_obj

    def cls_pred_total(self,dataroot):#“sub_obj":[int]
        en2cls = joinpath(dataroot,"entity_class.txt")
        fact_path = joinpath(dataroot,"fact.txt")
        # fact_dict = dict()
        # with open(fact_path) as f:
        #     for line in f:
        #         pred_id = line.strip().split()[1]
        #         sub,obj = line.strip().split()[0],line.strip().split()[2]
        #         fact_dict[sub+"_"+obj] = int(pred_id)
        fact_dict = dict()
        with open(fact_path) as f:
            for line in f:
                pred_id = line.strip().split()[1]
                sub, obj = line.strip().split()[0], line.strip().split()[2]
                if sub + "_" + obj not in fact_dict.keys():
                    fact_dict[sub + "_" + obj] = []
                    fact_dict[sub + "_" + obj].append(int(pred_id))
                else:
                    fact_dict[sub + "_" + obj].append(int(pred_id))
        with open(en2cls) as f:
            for line in f:
                ins,ins_id = line.strip().split(' ')
                fact_dict[ins] = int(ins_id)

        return fact_dict

    def find_rule(self,sub,ent2class):#sub是12_14之类的索引
        samples_by_r = []
        neg_mask_by_r = []
        obs_var_by_r = []
        latent_var_by_r = []
        r_index = []
        n=0
        for instance in sub:
            x,y,r = instance[0], instance[2], instance[1]
            instance_name='addition_'+ r
            x_name=ent2class.get(x)
            y_name=ent2class.get(y)
            if x_name==None:
                print("x",x)
            if y_name==None:
                print('y',y)
            for r_id,formula in enumerate(self.rule_ls):
                #print(formula)
                if x_name==formula.atom_ls[0].pred_name and y_name==formula.atom_ls[1].pred_name:
                    
                    ins = {'x':x,'y':y}
                    samples,neg_mask,latent_var,obs_var = self.get_rule_rnd(ins,formula)
                    samples_by_r.append(samples)
                    neg_mask_by_r.append(neg_mask)
                    obs_var_by_r.append(obs_var)
                    latent_var_by_r.append(latent_var)
                    r_index.append(r_id)
                    n=n+1
                    break    
                else:
                    continue
        #pdb.set_trace()
        return samples_by_r,neg_mask_by_r,obs_var_by_r,latent_var_by_r,r_index

    def get_rule_rnd(self,ins,rule):
        # x_real, y_real = ins
        # 按照规则里原子的个数赋予[]
        # sample_buff = [[] for _ in rule.atom_ls]  # []个数等于原子个数
        # neg_mask_buff = [[] for _ in rule.atom_ls]
        samples = [[atom.pred_name, []] for atom in rule.atom_ls]  # [规则对应的原子的谓词,[]] 把那条规则对应的原子集取出来
        neg_mask = [[atom.pred_name, []] for atom in rule.atom_ls] #记录！
        latent_mask = [[atom.pred_name, []] for atom in rule.atom_ls] #记录隐变量
        obs_var = [[atom.pred_name, []] for atom in rule.atom_ls] ##记录观测变量
        # neg_var = [[atom.pred_name, []] for atom in rule.atom_ls]
        #pdb.set_trace()
        for index,atom in enumerate(rule.atom_ls):
            if len(atom.var_name_ls) == 1:
                var = ins[atom.var_name_ls[0]]
                samples[index][1].append(int(var))
                latent_mask[index][1].append(1) # 1 represents latent variable
                # if var in self.mln_fact.keys() and atom.pred_name == self.pred_id2name[self.mln_fact[var]]:
                #     samples[index][1].append(int(var))
                #     obs_var[index][1].append(int(var))
                #     latent_mask[index][1].append(0)
                # else:
                #     samples[index][1].append(int(var))
                #     latent_mask[index][1].append(1)
                if atom.neg == 1:
                    neg_mask[index][1].append(0)
                else:
                    neg_mask[index][1].append(1)
            else:
                var = [ins[atom.var_name_ls[0]],ins[atom.var_name_ls[1]]]
                key = "_".join(var)
                figure_g = [int(ins[atom.var_name_ls[0]]),int(ins[atom.var_name_ls[1]])]
                samples[index][1].extend(figure_g)
                obs_var[index][1].extend(figure_g)
                latent_mask[index][1].append(0)
                # if key in self.mln_fact.keys():
                #     for rel in self.mln_fact[key]:
                #         if atom.pred_name == self.pred_id2name[rel]:
                #             figure_g = [int(ins[atom.var_name_ls[0]]),int(ins[atom.var_name_ls[1]])]
                #             samples[index][1].extend(figure_g)
                #             obs_var[index][1].extend(figure_g)
                #             latent_mask[index][1].append(0)
                #             break
                # else:
                #     figure_g = [int(ins[atom.var_name_ls[0]]),int(ins[atom.var_name_ls[1]])]
                #     samples[index][1].extend(figure_g)
                #     latent_mask[index][1].append(1)
                if atom.neg == 1:
                    neg_mask[index][1].append(0)
                else:
                    neg_mask[index][1].append(1)
        #pdb.set_trace()
        return samples,neg_mask,latent_mask,obs_var

    def get_batch_rnd(self,):#observed_prob=0.7, filter_latent=True, closed_world=False, filter_observed=False):

        samples_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        obs_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]

        cnt = 0
        inds = list(range(len(self.rule_ls)))

        # while cnt < self.batchsize:

        # 打乱规则顺序
        if self.shuffle_sampling:
            shuffle(inds)

        for ind in inds[:self.batchsize]:

            rule = self.rule_ls[ind]
            # print(rule)
            atom_key_dict = self.atom_key_dict_ls[ind]
            sub = [None]*len(rule.rule_vars)
            self._instantiate_pred(rule.atom_ls[:2],atom_key_dict,sub,rule)

            assert len(sub)==2

            # cnt += 1

            x_real,y_real = sub
            #按照规则里原子的个数赋予[]
            sample_buff = [[] for _ in rule.atom_ls]#[]个数等于原子个数
            neg_mask_buff = [[] for _ in rule.atom_ls]
            samples = samples_by_r[ind]#[规则对应的原子的谓词,[]] 把那条规则对应的原子集取出来
            neg_mask = neg_mask_by_r[ind]
            obs_var = obs_var_by_r[ind]
            neg_var = neg_var_by_r[ind]
            atom_init = 2#从第三个原子开始判断
            obs_var[0][1].append([sub[0]])#第一个原子的取值
            obs_var[1][1].append(sub)#第二个原子的取值
            sample_buff[0].append(x_real)
            sample_buff[1].extend(sub)
            neg_mask_buff[0].append(0)
            neg_mask_buff[1].append(0)
            for atom_ in rule.atom_ls[2:]:
                if (1,(str(y_real),)) in self.fact_dict[atom_.pred_name]:
                    obs_var[atom_init][1].append([y_real])
                else:
                    neg_var[atom_init][1].append([y_real])
                sample_buff[atom_init].append(y_real)
                neg_mask_buff[atom_init].append(1)
                atom_init += 1
            for i in range(len(rule.atom_ls)):
                samples[i][1].extend(sample_buff[i])
                neg_mask[i][1].extend(neg_mask_buff[i])

        return samples_by_r,neg_mask_by_r,obs_var_by_r,neg_var_by_r