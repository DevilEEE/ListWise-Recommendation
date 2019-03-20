# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:22:05 2019

@author: v_fdwang
"""

import torch as t

from torch.autograd import Variable as V
from model import Model, Generator

from graphviz import Digraph
import torch
from torch.autograd import Variable

config = {
        "dic_size": 200,
        "emb_size": 300,
        "max_push_len": 10,
        "lin1_size": 300,
        "out_size": 1,
        "use_cuda": False,
        "num_emb": 176,
        "hidden_dim": 300,
        "seed": 1234
        }


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t1 in var.saved_tensors:
                    dot.edge(str(id(t1)), str(id(var)))
                    add_nodes(t1)
    add_nodes(var.grad_fn)
    return dot

x = V(t.LongTensor([[1,2,3,2,1,0,0,0,0,0],[3,4,2,5,0,0,0,0,0,0]]))
model1 = Generator(config)
y = model1(x)
g = make_dot(y)
g.view()
