import torch
import torch.nn as nn

class Attention_Weights():
    def __init__(self, eps):
        self.eps = eps or 1e-3

    