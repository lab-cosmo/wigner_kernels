import torch

def get_mae(first, second):
    return torch.mean(torch.abs(first - second))

def get_sae(first, second):
    return torch.sum(torch.abs(first - second))

def get_rmse(first, second):
    return torch.sqrt(torch.mean((first - second)**2))

def get_sse(first, second):
    return torch.sum((first - second)**2)

def get_dipole_sae(first, second):
    first_magnitude = torch.sqrt((first.reshape(-1, 3) **2).sum(dim=1))
    second_magnitude = torch.sqrt((second.reshape(-1, 3) **2).sum(dim=1))
    return get_sae(first_magnitude, second_magnitude)

def get_dipole_mae(first, second):
    first_magnitude = torch.sqrt((first.reshape(-1, 3) **2).sum(dim=1))
    second_magnitude = torch.sqrt((second.reshape(-1, 3) **2).sum(dim=1))
    return get_mae(first_magnitude, second_magnitude)
