import torch.nn as nn
import torch
import math
import torch.nn.functional as F
class GraphSGConvolution(nn.Module):
    """
    A Graph SGC Layer with MLP
    """
    def __init__(self, in_channel, hidden_channel,  out_channel, bias=True, dropout = 0):
        super(GraphSGConvolution, self).__init__()
        self.in_features = in_channel
        self.hidden_channel  = hidden_channel
        self.out_channel = out_channel
        self.fcs = nn.ModuleList([nn.Linear(in_channel, hidden_channel, bias=bias)])
        self.fcs.append( nn.Linear(hidden_channel, out_channel, bias=bias))
        self.reg_params = list(self.fcs[0].parameters())    # l2 reg on the first linear layer
        self.dropout =  dropout
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.fcs[0].weight.size(1))
        stdv1 = 1. / math.sqrt(self.fcs[1].weight.size(1))
        self.fcs[0].weight.data.uniform_(-stdv, stdv)
        self.fcs[1].weight.data.uniform_(-stdv1, stdv1)


    def forward(self, input):
        """ f(AX)"""
        support_1 = self.fcs[0](input)
        support_1 = F.relu(support_1)
        support_1 = F.dropout(support_1, p = self.dropout, training =  True)

        support_2 = self.fcs[1](support_1)
        return support_2

    def error_rate(self, x, idx, labels):
        ## 错误率
        error_rate = torch.div(torch.sum(self.node_weight[idx] * (torch.argmax(x, dim = -1)[idx] != labels)),  torch.sum(self.node_weight)[idx])
        return error_rate

    def weighted_loss(self, x, idx, labels, node_weights):

        loss = nn.CrossEntropyLoss(reduction = 'none')(x[idx], labels)
        loss *= node_weights[idx]
        return loss.sum()/node_weights[idx].sum()