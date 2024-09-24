#LTN

import sys
# sys.path.append("..")
sys.path.append(r'/data/ydr2021/image_classification_CUB/AttentionZSL/')
import torch.nn as nn
import torch
import torch.nn.functional as F
from common.predicate import PRED_DICT
import json
import numpy as np
import pdb
class Posterior(nn.Module):

    def __init__(self,embed_dim,slice_dim,pred_txt):

        super(Posterior, self).__init__()

        self.embedding_dim = embed_dim
        self.device = 'cuda'
        self.rel2id = self.generate_rel2id(pred_txt)
        self.idx2rel = self.generate_id2rel(pred_txt)
        self.num_rels = len(self.rel2id)
        self.entropy = F.binary_cross_entropy_with_logits
        self.classify = nn.Softmax(dim=0)
        self.xent_loss = F.binary_cross_entropy_with_logits
        self.attr_list=self.generate_attribute()

        self.params_u_R = nn.ParameterList()
        self.params_W_R = nn.ModuleList()
        self.params_V_R = nn.ModuleList()
        self.params_b_R = nn.ParameterList()
        
        for idx in range(self.num_rels):
            
            rel = self.idx2rel[idx]
            
            #num_args = PRED_DICT[rel].num_args  # 统计是几元关系
            num_args =1
            self.params_u_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))
            self.params_W_R.append(nn.Bilinear(num_args * embed_dim, num_args * embed_dim, slice_dim, bias=False))
            self.params_V_R.append(nn.Linear(num_args * embed_dim, slice_dim, bias=False))
            self.params_b_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))
    def generate_attribute(self):
        attr_id_list =[]
        for key,value in self.rel2id.items():
            if value >=50:
                attr_id_list.append(value)
        return attr_id_list

    def generate_rel2id(self,path):
        rel = json.load(open(path))
        rel_dict = dict((name,rel.index(name)) for name in rel)
        return  rel_dict

    def generate_id2rel(self,path):
        rel = json.load(open(path))
        rel_dict = dict((rel.index(name),name) for name in rel)
        return rel_dict

    def forward(self,gt_dict,vars_gd,is_training=None):
        
        if is_training:
            
            samples, obs_mask, latent_mask = vars_gd
           
           
            #find attribute label id
            attr_id_list = []
            for id in samples:
             
                idx = self.rel2id[id[0]]
                attr_id_list.append(idx)
                
            #total_mask_mat里规则体部分是1（左侧），规则头部分是0（右侧） tensor.shape=[n,1]
            
            #1代表隐变量，0代表观测变量
            latent_mask_mat = torch.tensor([pred_mask[1] for pred_mask in latent_mask], dtype=torch.float).to(self.device)
            obs_mask_mat = (latent_mask_mat == 0).type(torch.float)
           
            x = samples[0][1]#ent1
         
            x_prob = self.compute_atom_pb(attr_id_list,x,gt_dict)# (ent1_id,embedding)
            
            #class_x= self.compute_classification(x_prob)#softmax
         
            x_prob = torch.stack(x_prob,dim=0)
            
            # l_mask = (latent_mask_mat==torch.ones_like(latent_mask_mat)).view(-1)
            # l=l_mask.cpu().numpy().tolist()
            
            obs_var = []
            a=0
            posterior_prob = torch.zeros(len(x_prob)).to(self.device)
            for index,value in enumerate(latent_mask_mat):#遍历观测变量
                
                if value.item()== 1.0:#跳过隐变量
                    a+=1
                    posterior_prob[index] = x_prob[index]
                    continue
                else:
                    posterior_prob[index] = 0.0
                    pred = samples[a][0]
                    pred_id = self.rel2id[pred]
                    a+=1
                    
                    #one_hot_label = torch.zeros(1, len(self.rel2id)).scatter_(1, torch.LongTensor([[pred_id,]]), 1)
                    
                    #ground_label = torch.cat(class_label, attr_label)
                    
                    #atom_loss += self.entropy(x_prob,one_hot_label,reduction="mean").cuda()
                    #pred_hinge = torch.tensor(x_prob).view(-1,1)
                    # print(pred_hinge[pred_id-70])
                    #hinge_loss += self.hinge_atom(torch.tensor(pred_hinge[pred_id-19].item()).cuda())
                    #class_x[index] = 1.0


                    #x_prob[index] = 1.0
                    obs_var.append(x_prob[index])
            #scores_all_pred = torch.stack(x_prob,dim=0).view(-1,1)
            
            # body_score = (1 - latent_mask_mat) * scores_all_pred
            # head_score = scores_all_pred *(1 - obs_mask_mat) 
           
            obs_mat = torch.stack(obs_var, dim=0)
            obs_loss = self.xent_loss(obs_mat, torch.ones_like(obs_mat), reduction='sum')
            
            obs_loss /= (obs_mat.size(0) + 1e-6)
            
            rule = 1-x_prob#rule body-1
            #rule = [1-x  for x in x_prob]
            rule[-1] = 1.0#rule head +1
            # # latent_score = self.compute_entropy(head_score[2:])
            # posterior_prob = (head_score+body_score)*latent_mask_mat
           

            entropy = self.compute_entropy(posterior_prob)
            
            #atoms score in a logic rule(离散的原子谓词)
            #and_score= (head_score+body_score)*latent_mask_mat+obs_mask_mat#将观测变量置1
            #transform disconjunction form a logic rule(such as !A ∨ !B ∨C)
            #or_score= (1 - total_mask_mat) * (1-and_score) + and_score *(1 - body_mask_mat) 
            
            
            or_score = rule

            potential = torch.mean(self.cal_rule(or_score))
            #print("规则得分  "+str(rule_score))
            
            return potential,entropy,x_prob, attr_id_list, obs_loss
        else:

            samples, test_name = vars_gd
            
            # a_sample_test_rule=[]
            # for one_rule in test_rule:
            #     #find attribute label id
            #     attr_id_list = []
            #     for id in one_rule:
                
            #         idx = self.rel2id[id[0]]
            #         attr_id_list.append(idx)
                
            #     x = one_rule[0][1]#ent1
                
            #     x_prob = self.compute_atom_pb(attr_id_list,x,gt_dict)# (ent1_id,embedding)
               
            #     x_prob = torch.stack(x_prob,dim=0)#grounding one rule, all atoms scores=[]

            #     rule = 1-x_prob#rule body-1
            #     rule[-1] = 1.0#rule head +1
            #     or_score = rule
            #     #potential = torch.mean(self.Goguen_rule(or_score)).item()
            #     potential = torch.mean(self.cal_rule(or_score)).item()
            #     a_sample_test_rule.append(potential)
               
            
            # max_value = max(a_sample_test_rule) # 求列表最大值
            # max_idx = a_sample_test_rule.index(max_value)
            # return  max_idx


            
            #find attribute label id
            attr_id_list = self.attr_list
            
            x = samples[0][1]
            
            x_prob = self.compute_atom_pb(attr_id_list,x,gt_dict)# (ent1_id,embedding)
            
            
            # pre_attr_label =[]
            # for i,pro in enumerate(x_prob):
            #     if pro >0.5:
            #        pro = 1
            #        pre_attr_label.append(pro)
            #     else:
            #        pro = 0
            #        pre_attr_label.append(pro)   
           
            # return  np.array(pre_attr_label)
            x_prob = torch.stack(x_prob,dim=0)
           
            return  np.array(x_prob.cpu())


    def compute_atom_pb(self,attr_id_list,pred_sample,node_embedd):
        
        if len(pred_sample) == 1:
            one_relation = []
            x_embd = torch.cat([node_embedd],dim=0)
            # probas = torch.zeros(len(attr_id_list)).cuda()
            # x_embd=node_embedd[pred_sample[0]]
            
            # for i in range(len(attr_id_list)):   
            #     rel_idx = attr_id_list[i]
            #     sample_score = self.params_u_R[rel_idx].dot(
            #         torch.tanh(self.params_W_R[rel_idx](x_embd, x_embd) +
            #                     self.params_V_R[rel_idx](x_embd) +
            #                     self.params_b_R[rel_idx])
            #     )
            #     proba = torch.sigmoid(sample_score)
                
            #     # probas[i] = proba
            #     one_relation.append(proba)
           
            for i in range(len(attr_id_list)):   
                rel_idx = attr_id_list[i]
                sample_score = self.params_u_R[rel_idx].dot(
                    torch.tanh(self.params_W_R[rel_idx](x_embd, x_embd) +
                                self.params_V_R[rel_idx](x_embd) +
                                self.params_b_R[rel_idx])
                )
                proba = torch.sigmoid(sample_score)
              
                one_relation.append(proba)


            
        return one_relation


    # def compute_classification(self,x_prob):

    #     tensor_result_x = torch.stack(x_prob,dim=0)
    #     result_x = self.classify(tensor_result_x)

    #     return result_x

    def compute_entropy(self,posterior_prob):
        eps = 1e-6
        posterior_prob.clamp_(eps, 1 - eps)
        compl_prob = 1 - posterior_prob
        entropy = -(posterior_prob * torch.log(posterior_prob) + compl_prob * torch.log(compl_prob)).sum()
        return entropy

    def F_Not(wff):
        # according to standard goedel logic is
        return 1-wff

    def F_Or(self,wffs):
        result = torch.sum(wffs, dim=0, keepdim=True)
        return torch.min(torch.ones_like(result, requires_grad=True), result)

    def F_And(self,wffs):
        result = torch.sum(wffs, dim=0, keepdim=False) 
        
        return torch.max(result, torch.zeros_like(result, requires_grad=True))

    def cal_rule(self,score):
        #按照t-norm计算规则
        s = score
        # print(s)
        for i in range(s.size()[0]-1):
            # print(s[i:i+2])

            temp = self.F_Or(s[i:i+2])

            s[i+1] = temp
        # print(temp)
        return temp
    def Goguen_rule(self,score):
        s = score
        for i in range(s.size()[0]-1):
            atom1, atom2 = s[i:i+2][0],s[i:i+2][1]
            temp = torch.mul(atom1, atom2)
            s[i+1] = temp
        return temp
    
    def hinge_atom(self,obs):
        result = 2 - 2*obs
        return torch.max(torch.zeros_like(result, requires_grad=True), result)
   # def hinge_atom(self,obs):
        #result = 1.2 - 2*obs
        #return torch.max(torch.zeros_like(result, requires_grad=True), result)