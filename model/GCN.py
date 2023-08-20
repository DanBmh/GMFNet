#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from model.Transformer import Transformer
from model.Scale import *

alpha = 0.2
lamda = 1.5


def make_quaternion_mul(kernel):

    """" The constructed 'hamilton' W is a modified version of the quaternion representation,

        thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """

    dim = kernel.size(1) // 8

    r, i, j, k, l, il, jl, kl = torch.split(kernel, [dim, dim, dim, dim, dim, dim, dim, dim], dim=1)

    r2 = torch.cat([r, -i, -j, -k, -l, -il, -jl, -kl], dim=0)  # 0,1,2,3,4,5,6,7

    i2 = torch.cat([i, r, -k, j, -il, l, kl, -jl], dim=0)  # 1,0,3,2,5,4,7,6

    j2 = torch.cat([j, k, r, -i, -jl, -kl, l, il], dim=0)  # 2,3,0,1,6,7,4,5

    k2 = torch.cat([k, -j, i, r, -kl, jl, -il, l], dim=0)  # 3,2,1,0,7,6,5,4

    l2 = torch.cat([l, il, jl, kl, r, -i, -j, -k], dim=0)  # 4,5,6,7,0,1,2,3

    il2 = torch.cat([il, -l, kl, -jl, i, r, -k, j], dim=0)  # 5,4,7,6,1,0,3,2

    jl2 = torch.cat([jl, -kl, -l, il, j, k, r, -i], dim=0)  # 6,7,4,5,2,3,0,1

    kl2 = torch.cat([kl, jl, -il, -l, k, -j, i, r], dim=0)  # 7,6,5,4,3,2,1,0

    hamilton = torch.cat([r2, i2, j2, k2, l2, il2, jl2, kl2], dim=1)

    assert kernel.size(1) == hamilton.size(1)

    return hamilton


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48, is_ga=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.is_ga = is_ga
        
        # --- add --- #
        if self.is_ga == True:
            self.weight_q = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features))

        self.reset_parameters()
        self.I = torch.eye(self.weight.shape[0])

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        # --- add --- #
        if self.is_ga == True:
            stdv = math.sqrt(6.0 / (self.weight_q.size(0) + self.weight_q.size(1)))
            self.weight_q.data.uniform_(-stdv, stdv)

    def forward(self, input, y0=None, layer_nums=0):
        
        # --- add --- #
        if self.is_ga == True:
            hamilton = make_quaternion_mul(self.weight_q)
        else:
            hamilton = self.weight

        if layer_nums != 0:
            # apply gcnii
            beta = math.log(lamda / layer_nums + 1)

            support = (1 - alpha) * torch.matmul(self.att, input) + alpha * y0
            # output = torch.matmul(support, beta * self.weight + (1 - beta) * self.I.to(self.weight.device))
            output = torch.matmul(support, beta * hamilton + (1 - beta) * self.I.to(self.weight.device))
        else:
            # support = torch.matmul(input, self.weight)
            support = torch.matmul(input, hamilton)
            output = torch.matmul(self.att, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48, layer_nums=0):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        if layer_nums % 2 == 1 or layer_nums > 8:

            self.flag = True

            self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, is_ga=True)
            self.bn1 = nn.BatchNorm1d(node_n * in_features)

            self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
            self.bn2 = nn.BatchNorm1d(node_n * in_features)

            self.do = nn.Dropout(p_dropout)
            self.act_f = nn.Tanh()

            self.trans = Transformer(d_input=256, d_model=256, d_output=256, n_layers=1, n_heads=8, d_k=32, d_v=32, d_ff=1024)

        else:

            self.flag = False

            self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, is_ga=True)
            self.bn1 = nn.BatchNorm1d(node_n * in_features)

            self.do = nn.Dropout(p_dropout)
            self.act_f = nn.Tanh()

            self.s1_2_s2 = AveargeJoint()
            self.s1_2_s3 = AveargePart()
            self.s2_2_s1 = PartLocalInform()
            self.s3_2_s1 = BodyLocalInform()

            self.gc_s1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
            self.gc_s2 = GraphConvolution(in_features, in_features, node_n=30, bias=bias)
            self.gc_s3 = GraphConvolution(in_features, in_features, node_n=15, bias=bias)
            self.bn_s1 = nn.BatchNorm1d(node_n * in_features)
            self.bn_s2 = nn.BatchNorm1d(30 * in_features)
            self.bn_s3 = nn.BatchNorm1d(15 * in_features)
            self.drop = nn.Dropout(p_dropout)

    def forward(self, x, y0=None, y0_s2=None, y0_s3=None, layer_nums=0):

        if self.flag == True:

            y = self.gc1(x, y0, layer_nums)
            b, n, f = y.shape
            y = self.bn1(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y)
            y = self.do(y)

            y = self.gc2(y, y0, layer_nums + 1)
            b, n, f = y.shape
            y = self.bn2(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y)
            y = self.do(y)

            y = self.trans(y)

            y = y + x

        else:

            y = self.gc1(x, y0, layer_nums)
            b, n, f = y.shape
            y = self.bn1(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y)
            y = self.do(y)

            y_s2 = self.s1_2_s2(y)
            y_s3 = self.s1_2_s3(y)
            y = self.gc_s1(y, y0)
            y_s2 = self.gc_s2(y_s2, y0_s2)
            y_s3 = self.gc_s3(y_s3, y0_s3)

            b, n, f = y.shape
            y = self.bn_s1(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y)
            y = self.drop(y)

            b, n, f = y_s2.shape
            y_s2 = self.bn_s2(y_s2.view(b, -1)).view(b, n, f)
            y_s2 = self.act_f(y_s2)
            y_s2 = self.drop(y_s2)

            b, n, f = y_s3.shape
            y_s3 = self.bn_s3(y_s3.view(b, -1)).view(b, n, f)
            y_s3 = self.act_f(y_s3)
            y_s3 = self.drop(y_s3)

            y_s21 = self.s2_2_s1(y_s2)
            y_s31 = self.s3_2_s1(y_s3)

            y = y + 0.3 * y_s21 + 0.3 * y_s31

        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gc1_s2 = GraphConvolution(input_feature, hidden_feature, node_n=30)
        self.bn1_s2 = nn.BatchNorm1d(30 * hidden_feature)

        self.gc1_s3 = GraphConvolution(input_feature, hidden_feature, node_n=15)
        self.bn1_s3 = nn.BatchNorm1d(15 * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n, layer_nums=i))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.s1_2_s2 = AveargeJoint()
        self.s1_2_s3 = AveargePart()
        self.s2_2_s1 = PartLocalInform()
        self.s3_2_s1 = BodyLocalInform()

    def forward(self, x):

        x_s2 = self.s1_2_s2(x)
        x_s3 = self.s1_2_s3(x)

        y = self.gc1(x)
        y_s2 = self.gc1_s2(x_s2)
        y_s3 = self.gc1_s3(x_s3)
        y0 = y
        y0_s2 = y_s2
        y0_s3 = y_s3

        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        b, n, f = y_s2.shape
        y_s2 = self.bn1_s2(y_s2.view(b, -1)).view(b, n, f)
        y_s2 = self.act_f(y_s2)
        y_s2 = self.do(y_s2)

        b, n, f = y_s3.shape
        y_s3 = self.bn1_s3(y_s3.view(b, -1)).view(b, n, f)
        y_s3 = self.act_f(y_s3)
        y_s3 = self.do(y_s3)

        y_s21 = self.s2_2_s1(y_s2)
        y_s31 = self.s3_2_s1(y_s3)

        y = y + 0.3 * y_s21 + 0.3 * y_s31

        for i in range(self.num_stage):
            y = self.gcbs[i](y, y0, y0_s2, y0_s3, 2 * i + 1)

        y = self.gc7(y)
        y = y + x

        return y
