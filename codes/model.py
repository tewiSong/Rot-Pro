#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, args,hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.train_pr_prop = args.train_pr

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-(self.embedding_range).item(),
            b=(self.embedding_range).item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item()* args.init_pr,
            b=self.embedding_range.item()* args.init_pr
        )

        # for RotPro:
        if self.model_name == 'RotPro':
            # self.projection_embedding_a = nn.Parameter(
            #     torch.ones(nrelation, self.relation_dim),
            #     requires_grad=False)
            self.projection_embedding_a = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.projection_embedding_a,
                a=0.5,
                b=0.5
            )

            self.projection_embedding_b = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.projection_embedding_b,
                a=0.5,
                b=0.5
            )


        if self.model_name == 'RotPro':
            self.projection_phase = nn.Parameter(torch.zeros(nrelation, self.relation_dim))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'RotPro']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            if self.model_name == 'RotPro':
                proj_a = torch.index_select(
                    self.projection_embedding_a,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)

                proj_b = torch.index_select(
                    self.projection_embedding_b,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)
          
            if self.model_name == 'RotPro':
                proj_p = torch.index_select(
                    self.projection_phase,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            if self.model_name == 'RotPro':
                proj_a = torch.index_select(
                    self.projection_embedding_a,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

                proj_b = torch.index_select(
                    self.projection_embedding_b,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

            

            if self.model_name == 'RotPro':
                proj_p = torch.index_select(
                    self.projection_phase,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if self.model_name == 'RotPro':
                proj_a = torch.index_select(
                    self.projection_embedding_a,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)

                proj_b = torch.index_select(
                    self.projection_embedding_b,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)


            if self.model_name == 'RotPro':
                proj_p = torch.index_select(
                    self.projection_phase,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        elif self.model_name == 'RotPro':
            score = self.RotPro(head, relation, tail, proj_a, proj_b, proj_p, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)  
        
        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)  
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail 

            re_score = re_score - re_head
            im_score = im_score - im_head  

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)  
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=2)
        return score

    def RotPro(self, head, relation, tail, proj_a, proj_b, proj_p, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)  

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = (relation / (self.embedding_range.item() / pi) )  * self.train_pr_prop

        phase_projection = proj_p

        re_projection = torch.cos(phase_projection)
        im_projection = torch.sin(phase_projection)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)


        ma = re_projection * re_projection * proj_a + im_projection * im_projection * proj_b
        mb = re_projection * im_projection * (proj_b - proj_a)
        md = re_projection * re_projection * proj_b + im_projection * im_projection * proj_a
        # p(et)
        re_tail_proj = ma * re_tail + mb * im_tail
        im_tail_proj = mb * re_tail + md * im_tail


        # p(eh)
        re_head_proj = ma * re_head + mb * im_head
        im_head_proj = mb * re_head + md * im_head


        if mode == 'head-batch':

            re_score = re_relation * re_tail_proj + im_relation * im_tail_proj
            im_score = re_relation * im_tail_proj - im_relation * re_tail_proj
            re_score = re_score - re_head_proj
            im_score = im_score - im_head_proj


        else:
            re_score = re_head_proj * re_relation - im_head_proj * im_relation
            im_score = re_head_proj * im_relation + im_head_proj * re_relation
            re_score = re_score - re_tail_proj
            im_score = im_score - im_tail_proj


        score = torch.stack([re_score, im_score],dim=0)  
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=2)

   
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)


        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
    
        negative_score = model((positive_sample, negative_sample), mode=mode)
        
        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
            
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        
        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)



        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
         

        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
           


        loss = (positive_sample_loss + negative_sample_loss) / 2
       


        if model.model_name == 'RotPro' and args.constrains:
            a1 = model.projection_embedding_a - 1.0
            a0 = model.projection_embedding_a - 0.0
            a = torch.abs(a1 * a0)
            penalty = torch.ones_like(a)
            penalty[a > args.gamma_m] = args.beta
            l_a = (a * penalty).norm(p=2)

            b1 = model.projection_embedding_b - 1.0
            b0 = model.projection_embedding_b - 0.0
            b = torch.abs(b1 * b0)
            penalty = torch.ones_like(b)
            penalty[b > args.gamma_m] = args.beta
            l_b = (b * penalty).norm(p=2)
            loss += (l_a + l_b) * args.alpha
            # loss += l_b * args.alpha

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
          
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        # print(positive_sample)
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            # print(ranking)
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
