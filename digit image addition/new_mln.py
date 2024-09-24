#LTN

import sys
# sys.path.append("..")
sys.path.append(r'/home/yudongran/ydr2/general_framework/')
import torch.nn as nn
import torch
import torch.nn.functional as F
from common.predicate import PRED_DICT
import json
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

        self.params_u_R = nn.ParameterList()
        self.params_W_R = nn.ModuleList()
        self.params_V_R = nn.ModuleList()
        self.params_b_R = nn.ParameterList()
        
        for idx in range(self.num_rels):
            rel = self.idx2rel[idx]
            num_args = PRED_DICT[rel].num_args  # 统计是几元关系
            self.params_u_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))
            self.params_W_R.append(nn.Bilinear(num_args * embed_dim, num_args * embed_dim, slice_dim, bias=False))
            self.params_V_R.append(nn.Linear(num_args * embed_dim, slice_dim, bias=False))
            self.params_b_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))


    def generate_rel2id(self,path):
        rel = json.load(open(path))
        rel_dict = dict((name,rel.index(name)) for name in rel)
        return  rel_dict

    def generate_id2rel(self,path):
        rel = json.load(open(path))
        rel_dict = dict((rel.index(name),name) for name in rel)
        return rel_dict

    def forward(self,gt_dict,vars_gd,is_training=True):
    #def forward(self,vars_gd,is_training=True):
        if is_training:
            samples, neg_mask, obs_var, latent_mask = vars_gd
            scores = []
            obs_probs = []
            neg_probs = []
            #pdb.set_trace()
            #total_mask_mat里规则体部分是0（左侧），规则头部分是1（右侧） tensor.shape=[n,1]
            total_mask_mat = torch.tensor([pred_mask[1] for pred_mask in neg_mask], \
                                          dtype=torch.float).to(self.device)
            body_mask_mat = (total_mask_mat == 0).type(torch.float)
            #1代表取隐变量
            latent_mask_mat = torch.tensor([pred_mask[1] for pred_mask in latent_mask], dtype=torch.float).to(self.device)
            obs_mask_mat = (latent_mask_mat == 0).type(torch.float)
            #pdb.set_trace()
            #print ('samples',samples)
            x = samples[0][1]#ent1
            y = samples[1][1]#ent2
            pred = samples[2][1]#relation
            # index = samples[1][1]
            # sub = torch.tensor(samples[1][1],dtype=torch.long).to(self.device)#[1,2]
            #pdb.set_trace()
            x_prob = self.compute_atom_pb(x,gt_dict)# (ent1_id,embedding)
            y_prob = self.compute_atom_pb(y,gt_dict)
            pred_prob = self.compute_atom_pb(pred,gt_dict)# (ent2_id,embedding)
            class_x,class_y,class_pred = self.compute_classification(x_prob,y_prob,pred_prob)
            scores.append(x_prob[self.rel2id[samples[0][0]]-19])
            scores.append(y_prob[self.rel2id[samples[1][0]]-19])
            scores.append(pred_prob[self.rel2id[samples[2][0]]])
            predicate_socre = pred_prob[self.rel2id[samples[2][0]]]


            # for ind in range(len(samples)):
            #     if ind < 2:
            #         continue
            #     pred_index = self.rel2id[samples[ind][0]]-19
            #     scores.append(y_prob[pred_index])
            #print(scores)

            l_mask = (latent_mask_mat==torch.ones_like(latent_mask_mat)).view(-1)
            l=l_mask.cpu().numpy().tolist()
            atom_loss = 0.0
            hinge_loss = 0.0
            
            # for i in range(l_mask.size()[0]):#遍历观测变量
            #     if i == True:#跳过隐变量
            #         continue
            a=0
            #pdb.set_trace()
            for j,i in enumerate(l):#遍历观测变量
                if i == 1:#跳过隐变量
                    a+=1
                    continue
                else:
                    
                    pred = samples[a][0]
                    pred_id = self.rel2id[pred]
                    a+=1
                    if pred_id < 19:#说明二元谓词是观测变量
                        pred_score_mat = torch.tensor(pred_prob).view(1,-1)
                        pred_one_hot = torch.zeros(1, 19).scatter_(1, torch.LongTensor([[pred_id,]]), 1)
                        atom_loss += self.entropy(pred_score_mat,pred_one_hot,reduction="mean").cuda()#只用观察变量训练推断网络
                        pred_hinge = torch.tensor(pred_prob).view(-1,1)
                        # print(pred_hinge[pred_id])
                        hinge_loss += self.hinge_atom(torch.tensor(pred_hinge[pred_id].item()).cuda())
                        #print('hinge_loss',hinge_loss)
                    else:
                        if j < 1:
                            x_score_mat = torch.tensor(x_prob).view(1,-1)
                            x_one_hot = torch.zeros(1, 10).scatter_(1, torch.LongTensor([[pred_id-19,]]), 1)
                            atom_loss += self.entropy(x_score_mat,x_one_hot,reduction="mean").cuda()
                            pred_hinge = torch.tensor(x_prob).view(-1,1)
                            # print(pred_hinge[pred_id-70])
                            hinge_loss += self.hinge_atom(torch.tensor(pred_hinge[pred_id-19].item()).cuda())
                            #print('hinge_loss', hinge_loss)
                        else:
                            y_score_mat = torch.tensor(y_prob).view(1,-1)
                            y_one_hot = torch.zeros(1, 10).scatter_(1, torch.LongTensor([[pred_id-19,]]), 1)
                            atom_loss += self.entropy(y_score_mat,y_one_hot,reduction="mean").cuda()
                            pred_hinge = torch.tensor(y_prob).view(-1,1)
                            # print(pred_hinge[pred_id-70])
                            hinge_loss += self.hinge_atom(torch.tensor(pred_hinge[pred_id-19].item()).cuda())
                            #print('hinge_loss', hinge_loss)


            scores_all_pred = torch.stack(scores,dim=0).view(-1,1)
            scores_all_pred = torch.sigmoid(scores_all_pred)
            # print(scores_all_pred)
            #pdb.set_trace()
            body_score = (1 - total_mask_mat) * scores_all_pred
            head_score = scores_all_pred *(1 - body_mask_mat) 

            # latent_score = self.compute_entropy(head_score[2:])
            latent_ = (head_score+body_score)*latent_mask_mat
            latent_score = self.compute_entropy(latent_)
            #atoms score in a logic rule(离散的原子谓词)
            and_score= (head_score+body_score)*latent_mask_mat+obs_mask_mat#将观测变量置1
            #transform disconjunction form a logic rule(such as !A ∨ !B ∨C)
            or_score= (1 - total_mask_mat) * (1-and_score) + and_score *(1 - body_mask_mat) 
            # print('and_score '+str(and_score))
            # print(and_score.size())
            # neg_score.shape[0] + 1e-6)
            rule_score = self.cal_rule(or_score)
            #print("规则得分  "+str(rule_score))
            
            return rule_score,atom_loss,hinge_loss,class_x,class_y,class_pred,latent_score,predicate_socre,latent_

    def compute_atom_pb(self,pred_sample,node_embedd):
        # if type(pred_sample) == int:
        #pdb.set_trace()
        if len(pred_sample) == 1:
            one_relation = []
            #pdb.set_trace()
            
            x_embd=node_embedd[str(pred_sample[0])]
            x_embd=x_embd.reshape(x_embd.size()[0]*x_embd.size()[1]*x_embd.size()[2])
            #x_embd = torch.cat([emdd],dim=0)
            #x_embd = torch.cat(node_embedd,dim=0)
            for pred_name,P in PRED_DICT.items():
                if P.num_args == 1:
                    rel_idx = self.rel2id[pred_name]
                    # self.params_u_R[rel_idx].to(self.device)
                    # self.params_W_R[rel_idx].to(self.device)
                    # self.params_V_R[rel_idx].to(self.device)
                    # sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](x_embd, x_embd) +
                    #                                                    self.params_V_R[rel_idx](x_embd))).view(-1)
                    sample_score = self.params_u_R[rel_idx].dot(
                        torch.tanh(self.params_W_R[rel_idx](x_embd, x_embd) +
                                   self.params_V_R[rel_idx](x_embd) +
                                   self.params_b_R[rel_idx])
                    )
                    one_relation.append(sample_score)
                else:
                    continue
            assert len(one_relation) == 10
            #print('one_relation_score',one_relation)
            return one_relation
        else:
            two_relation = []
            emdd1=node_embedd[str(pred_sample[0])].reshape(256)
            emdd2=node_embedd[str(pred_sample[1])].reshape(256)
            joint_embed = torch.cat((emdd1,emdd2),dim=0)
            
            for pred_name,P in PRED_DICT.items():
                if P.num_args == 2:
                    rel_idx = self.rel2id[pred_name]
                    # self.params_u_R[rel_idx].to(self.device)
                    # self.params_W_R[rel_idx].to(self.device)
                    # self.params_V_R[rel_idx].to(self.device)
                    # sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](joint_embed, joint_embed) +
                    #                                                    self.params_V_R[rel_idx](joint_embed))).view(-1)
                    sample_score = self.params_u_R[rel_idx].dot(
                        torch.tanh(self.params_W_R[rel_idx](joint_embed, joint_embed) +
                                   self.params_V_R[rel_idx](joint_embed) +
                                   self.params_b_R[rel_idx])
                    )
                    two_relation.append(sample_score)
                else:
                    continue
            assert len(two_relation) == 19
            return two_relation

    def compute_classification(self,x_prob,y_prob,pred_prob):

        # tensor_result_x = torch.sigmoid(torch.stack(x_prob,dim=0))
        # tensor_result_y = torch.sigmoid(torch.stack(y_prob,dim=0))
        # tensor_result_p = torch.sigmoid(torch.stack(pred_prob,dim=0))
        
        tensor_result_x = torch.stack(x_prob,dim=0)
        tensor_result_y = torch.stack(y_prob,dim=0)
        tensor_result_p = torch.stack(pred_prob,dim=0)
        
        result_x = self.classify(tensor_result_x)
        result_y = self.classify(tensor_result_y)
        result_p = self.classify(tensor_result_p)

        return result_x,result_y,result_p

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
        s = score.clone()
        # print(s)
        for i in range(s.size()[0]-1):
            # print(s[i:i+2])

            temp = self.F_Or(s[i:i+2])

            s[i+1] = temp
        # print(temp)
        return temp

    def hinge_atom(self,obs):
        result = 2 - 2*obs
        return torch.max(torch.zeros_like(result, requires_grad=True), result)
   # def hinge_atom(self,obs):
        #result = 1.2 - 2*obs
        #return torch.max(torch.zeros_like(result, requires_grad=True), result)