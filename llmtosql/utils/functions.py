import torch


def custom_pre_process_function():
    pass


def custom_loss_function(output, target):
    loss = torch.mean((output - target)**2)
    return loss
