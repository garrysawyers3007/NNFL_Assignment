import torch
import torch.nn as nn

class L1Criterion():
    
    def __init__(self, alpha):
        self.alpha = alpha
    
    def updateOutput(input):
        self.output = self.alpha*torch.norm(input, 1)
        return self.output

    def updateGradInput(input):
        self.gradInput = torch.mul(torch.sign(input), self.alpha)
        return self.gradInput