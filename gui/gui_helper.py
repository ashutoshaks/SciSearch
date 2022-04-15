import math
import json
from time import time
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

# Learn semantic relatedness between two feature vectors
class SemRelNN (nn.Module):
    def __init__(self, max_score=3):
        super(SemRelNN, self).__init__()
        self.classes = max_score + 1
        self.dense = nn.Linear(1, self.classes, bias=True)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, ins1, ins2):
        inputs = ins1 * ins2
        inputs = inputs.sum(axis=1)
        inputs = inputs.reshape((-1, 1))
        logits = self.dense(inputs)
        dist = self.soft(logits)
        return dist

# Transfer learning from the pre-trained models
class FineTuneNN (nn.Module):
    def __init__(self, input_dim, output_dim=512, act=torch.tanh):
        super(FineTuneNN, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.dense1 = nn.Linear(input_dim, output_dim, bias=True)
        self.dense2 = nn.Linear(output_dim, output_dim, bias=True)
        self.act_fn = act

    def forward(self, inputs):
        out1 = self.dense1(inputs)
        act_out1 = self.act_fn(out1)
        out2 = self.dense2(act_out1)
        act_out2 = self.act_fn(out2)
        return act_out2

class QBESciAR (nn.Module):
    def __init__(self, baseline, facet, input_dim, max_score=3, act=torch.tanh):
        super(QBESciAR, self).__init__()
        self.baseline = baseline
        self.facet = facet
        self.fine_tuner = FineTuneNN(input_dim, input_dim, act)
        self.score_gen = SemRelNN(max_score)
        self.train_acc_step = []
        self.train_acc_epoch = []
        self.train_loss_step = []
        self.train_loss_epoch = []
        self.valid_acc = []
        self.valid_loss = []

    def forward(self, ins1, ins2):
        tuned1 = self.fine_tuner(ins1)
        tuned2 = self.fine_tuner(ins2)
        return self.score_gen(tuned1, tuned2)

def GetQueryFeature(baseline, facet, paper_id):
    query_feature_data_file_name = '../data/' + baseline + '/' + facet + '.json'
    query_feature_data_file = open(query_feature_data_file_name)
    query_feature_data = json.load(query_feature_data_file)
    query_feature_data_file.close()
    return query_feature_data[paper_id]


def GetCandidateFeatures(baseline, facet, paper_id):
    rank_data_file_name = '../data/test-pid2anns-csfcube-' + facet + '.json'
    rank_data_file = open(rank_data_file_name)
    rank_data = json.load(rank_data_file)

    cand_feature_data_file_name = '../data/' + baseline + '/all.json'
    cand_feature_data_file = open(cand_feature_data_file_name)
    cand_feature_data = json.load(cand_feature_data_file)

    rank_data_file.close()
    cand_feature_data_file.close()

    cand_features = [cand_feature_data[cand_pid]
                     for cand_pid in rank_data[paper_id]['cands']]
    return rank_data[paper_id]['cands'], cand_features

def QBERetrieveSciArticles(baseline, facet, paper_id, loss_fn = "KLDivLoss", top=True, ret_k=15):
    facet = facet.lower()
    qf = GetQueryFeature(baseline, facet, paper_id)
    cand_pids, cand_f = GetCandidateFeatures(baseline, facet, paper_id)
    cand_c = len(cand_f)

    model_name = baseline + '/' + facet + '.qbe'
    model = torch.load('../models-' + loss_fn + '/' + model_name)
    model.eval()

    ins1, ins2 = [qf] * cand_c, cand_f
    ins1, ins2 = torch.tensor(ins1).float(), torch.tensor(ins2).float()

    pred = model(ins1, ins2)
    pred_labels = torch.argmax(pred, dim=1).numpy()
    pred = pred.detach().numpy()
    rank_scores = [l + (l != 0) * pred[i][l] + (l == 0) * (1 - pred[i][0])
                   for i, l in enumerate(pred_labels)]

    scored_cands = list(zip(cand_pids, rank_scores))
    scored_cands.sort(key=lambda x: -1 * x[1])

    return scored_cands[:ret_k]
