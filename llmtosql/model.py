from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.nn.parameter import Parameter
import structlog
import os

logger = structlog.get_logger('__name__')


class WikiSQLModel(nn.Module):
    def __init__(self):
        super().__init__()




    def forward(self, data):
        pass
