import torch
import torch.nn as nn
from common.cmd_args import cmd_args
from common.predicate import PRED_DICT
import torch.nn.functional as F
import json
import pdb
class FactorizedPosterior(nn.Module):
    def __init__(self, latent_dim, slice_dim=5):
        super(FactorizedPosterior, self).__init__()

        #self.graph = graph
        self.latent_dim = latent_dim

        self.xent_loss = F.binary_cross_entropy_with_logits

        self.device = 'cuda'
        #self.max_t_norm = self.max_t_norm

        
        self.rel2idx = self.generate_rel2id('/data/ydr2021/image_classification_CUB/AttentionZSL/data/CUB/mln/predicates.json')
        self.idx2rel = self.generate_id2rel('/data/ydr2021/image_classification_CUB/AttentionZSL/data/CUB/mln/predicates.json')
        self.num_rels = len(self.rel2idx)
        self.ent2idx = self.generate_ent2id('/data/ydr2021/image_classification_CUB/AttentionZSL/data/CUB/mln/ent2class.txt',self.rel2idx)
        
        

        self.params_u_R = nn.ModuleList()
        self.params_W_R = nn.ModuleList()
        self.params_V_R = nn.ModuleList()
        for idx in range(self.num_rels):
           
            rel = self.idx2rel[idx]
            num_args = PRED_DICT[rel].num_args
            self.params_W_R.append(nn.Bilinear(num_args * latent_dim, num_args * latent_dim, slice_dim, bias=False))
            self.params_V_R.append(nn.Linear(num_args * latent_dim, slice_dim, bias=True))
            self.params_u_R.append(nn.Linear(slice_dim, 1, bias=False))
        
        # self.params_u_R = nn.ParameterList()
        # self.params_W_R = nn.ModuleList()
        # self.params_V_R = nn.ModuleList()
        # self.params_b_R = nn.ParameterList()
        # for idx in range(self.num_rels):
        #     rel = self.idx2rel[idx]
           
        #     num_args = PRED_DICT[rel].num_args
        #     self.params_u_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))
        #     self.params_W_R.append(nn.Bilinear(num_args * latent_dim, num_args * latent_dim, slice_dim, bias=False))
        #     self.params_V_R.append(nn.Linear(num_args * latent_dim, slice_dim, bias=False))
        #     self.params_b_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))
    
 
    def generate_ent2id(self,ent_path,rel2id):
        ent2id = {}
       
        with open(ent_path) as f:
            for line in f:
               
                ent2id[line.strip( ).split()[0]]= rel2id[line.strip( ).split()[1]]
       
        return ent2id
    
    def generate_rel2id(self,path):
        rel = json.load(open(path))
        rel_dict = dict((name,rel.index(name)) for name in rel)
        return  rel_dict

    def generate_id2rel(self,path):
        rel = json.load(open(path))
        rel_dict = dict((rel.index(name),name) for name in rel)
        return rel_dict
    
    # def max_t_norm(self,rule_score):
    #    # Element-wise maximum of two tensors
    #    for i in range(len(rule_score)):
    #        if i == 0:
    #           x=0
    #           y=rule_score[i]
    #           x = torch.max(x, y)
    #        else:
    #            y = rule_score[i]
    #            x = torch.max(x, y)
    #    return x
    ############################################带观察变量
    # def forward(self, latent_vars, node_embeds,training = None):
    #     if training:
    #         node_embeds = node_embeds.reshape(1, node_embeds.shape[0])
    #         samples, latent_mask, obs_mask,obs_var, neg_mask = latent_vars
    #         scores = []
    #         obs_probs = []
    #         neg_probs = []

    #         pos_mask_mat = torch.tensor([pred_mask[1] for pred_mask in neg_mask], dtype=torch.float).to(self.device)
    #         neg_mask_mat = (pos_mask_mat == 0).type(torch.float)
    #         latent_mask_mat = torch.tensor([pred_mask[1] for pred_mask in latent_mask], dtype=torch.float).to(self.device) #hidden varible =1
    #         obs_mask_mat = (latent_mask_mat == 0).type(torch.float)# observed variable = 1
            
    #         for ind in range(len(samples)):
    #             pred_name, pred_sample = samples[ind]
    #             _, obs_sample = obs_var[ind]
    #             #_, neg_sample = neg_var[ind]

    #             rel_idx = self.rel2idx[pred_name]

    #             sample_mat = torch.tensor(pred_sample, dtype=torch.long).to(self.device)
    #             obs_mat = torch.tensor(obs_sample, dtype=torch.long).to(self.device)
              
    #             sample_mat = torch.cat([sample_mat, obs_mat], dim=0)
                
    #             if len(sample_mat) !=1:
    #                sample_query = torch.cat([node_embeds, node_embeds], dim=0)
    #             else:
    #                 sample_query = torch.cat([node_embeds, []], dim=0)
                
    #             sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](sample_query,sample_query) +
    #                                                         self.params_V_R[rel_idx](sample_query))).view(-1)
            
    #             var_prob = sample_score[len(pred_sample):]
    #             obs_prob = var_prob[:len(obs_sample)]
               
    #             sample_score = sample_score[:len(pred_sample)]

    #             scores.append(sample_score)
    #             obs_probs.append(obs_prob)
               
          
    #         score_mat = torch.stack(scores, dim=0)
    #         score_mat = torch.sigmoid(score_mat)
            
    #         pos_score = (1 - score_mat) * pos_mask_mat
    #         neg_score = score_mat * neg_mask_mat

    #         #potential = 1 - ((pos_score + neg_score) * latent_mask_mat + obs_mask_mat).prod(dim=0)
    #         atoms_score = (pos_score + neg_score) * latent_mask_mat + obs_mask_mat
            
    #         for i in range(len(atoms_score)):
    #             if i == 0:
    #                 x=torch.zeros(1).to(self.device)
    #                 y=atoms_score[i]
    #                 x = torch.max(x, y)
    #             else:
    #                 y = atoms_score[i]
    #                 x = torch.max(x, y)
    #         potential = x
    #         obs_mat = torch.cat(obs_probs, dim=0)
            
    #         if obs_mat.size(0) == 0:
    #             obs_loss = 0.0
    #         else:
    #             obs_loss = self.xent_loss(obs_mat, torch.ones_like(obs_mat), reduction='sum')

            
    #         obs_loss /= (obs_mat.size(0) + 1e-6)
           
    #         return potential, (score_mat * latent_mask_mat).view(-1), obs_loss, score_mat[:-1].view(-1)
        ############################################不带观察变量
    def forward(self, latent_vars, node_embeds,training = None):
        if training:
            node_embeds = node_embeds.reshape(1, node_embeds.shape[0])
            samples, latent_mask, obs_mask,obs_var, neg_mask = latent_vars
            scores = []
            obs_probs = []
            neg_probs = []

            pos_mask_mat = torch.tensor([pred_mask[1] for pred_mask in neg_mask], dtype=torch.float).to(self.device)
            neg_mask_mat = (pos_mask_mat == 0).type(torch.float)
            latent_mask_mat = torch.tensor([pred_mask[1] for pred_mask in latent_mask], dtype=torch.float).to(self.device) #hidden varible =1
            obs_mask_mat = (latent_mask_mat == 0).type(torch.float)# observed variable = 1
            
            for ind in range(len(samples)):
                pred_name, pred_sample = samples[ind]
                #_, obs_sample = obs_var[ind]
                #_, neg_sample = neg_var[ind]

                rel_idx = self.rel2idx[pred_name]

                sample_mat = torch.tensor(pred_sample, dtype=torch.long).to(self.device)
                #obs_mat = torch.tensor(obs_sample, dtype=torch.long).to(self.device)
              
                #sample_mat = torch.cat([sample_mat, obs_mat], dim=0)
                
                # if len(sample_mat) !=1:
                #    sample_query = torch.cat([node_embeds, node_embeds], dim=0)
                # else:
                #     sample_query = torch.cat([node_embeds, []], dim=0)
                sample_query = node_embeds
                
                sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](sample_query,sample_query) +
                                                            self.params_V_R[rel_idx](sample_query))).view(-1)
            
                # var_prob = sample_score[len(pred_sample):]
                # obs_prob = var_prob[:len(obs_sample)]
               
                # sample_score = sample_score[:len(pred_sample)]

                scores.append(sample_score)
                #obs_probs.append(obs_prob)
               
          
            score_mat = torch.stack(scores, dim=0)
            score_mat = torch.sigmoid(score_mat)
            
            pos_score = (1 - score_mat) * neg_mask_mat
            neg_score = score_mat * pos_mask_mat
           
            #potential = 1 - ((pos_score + neg_score) * latent_mask_mat + obs_mask_mat).prod(dim=0)
            
            atoms_score = (pos_score + neg_score) * latent_mask_mat 
            
            for i in range(len(atoms_score)):
                if i == 0:
                    x=torch.zeros(1).to(self.device)
                    y=atoms_score[i]
                    x = torch.max(x, y)
                else:
                    y = atoms_score[i]
                    x = torch.max(x, y)
            potential = x
            #obs_mat = torch.cat(obs_probs, dim=0)
           
            # if obs_mat.size(0) == 0:
            #     obs_loss = 0.0
            # else:
            #     obs_loss = self.xent_loss(obs_mat, torch.ones_like(obs_mat), reduction='sum')

            
            # obs_loss /= (obs_mat.size(0) + 1e-6)
            obs_loss = 0.0
            return potential, (score_mat * latent_mask_mat).view(-1), obs_loss, score_mat[:-1].view(-1)
        ##########################################################################################
        # else:
        #     node_embeds.reshape(1, node_embeds.shape[0])
        #     samples, test_rules = latent_vars
            
        #     all_test_rules_score = []
        #     for rule in test_rules:#计算一个实例grounding 所有测试规则的评分
        #         all_atoms_scores = []
        #         for ind in range(len(rule)-1):
                    
        #             name, sample = rule[ind]
                
        #             rel_idx = self.rel2idx[name]

        #             sample_query = node_embeds

        #             sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](sample_query, sample_query) +
        #                                                             self.params_V_R[rel_idx](sample_query))).view(-1) # (bsize)
        #             all_atoms_scores.append(torch.sigmoid(sample_score))
               
        #         one_rule_score = torch.stack(all_atoms_scores, dim=0)
              
        #         pre_rulebody_scores = one_rule_score.prod(dim=0) 
                
        #         all_test_rules_score.append(pre_rulebody_scores)
            
        #     all_test_rules_score = torch.stack(all_test_rules_score, dim=0).view(-1)
      
        #     return all_test_rules_score
        ##############################################################
        else:
            node_embeds.reshape(1, node_embeds.shape[0])
            samples = latent_vars
            scores = []
            
            for ind in range(len(samples[0])-1):#only compute rule body (attributes)
                name, sample = samples[0][ind]
               
                rel_idx = self.rel2idx[name]

                sample_query = node_embeds

                sample_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](sample_query, sample_query) +
                                                                self.params_V_R[rel_idx](sample_query))).view(-1) # (bsize)
                scores.append(torch.sigmoid(sample_score))
            
            one_rule_score = torch.stack(scores, dim=0)
            trans_conjunction = 1.0 - one_rule_score
            
            pre_rulebody_scores = torch.max(one_rule_score)
            
            #return torch.sigmoid(pre_rule_scores)
            return pre_rulebody_scores