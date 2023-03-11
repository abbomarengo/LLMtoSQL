import torch

criterion = torch.nn.CrossEntropyLoss()

def custom_pre_process_function():
    pass


def custom_loss_function(outputs, targets):
    losses = [criterion(output, target) for output, target in zip(outputs, targets)]
    loss = torch.stack(losses, dim=0).sum(dim=0)
    return loss
