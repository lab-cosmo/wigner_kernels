import numpy as np
import torch

class ValidationCycle(torch.nn.Module):
    # Evaluates the model on the validation set so that derivatives 
    # of an arbitrary loss with respect to the continuous
    # hyperparameters can be used to minimize the validation loss.

    def __init__(self, nu_max, alpha_exp):
        super().__init__()

        # Kernel regularization:
        self.sigma_exponent = torch.nn.Parameter(
            torch.tensor([alpha_exp], dtype = torch.get_default_dtype())
            )

        # Coefficients for mixing kernels of different body-orders:
        self.coefficients = torch.nn.Linear(nu_max+1, 1, bias = False)
        self.coefficients.weight = torch.nn.Parameter(
            torch.concat([torch.zeros((1,)), torch.zeros((nu_max,))]).reshape(1, -1)
        )
        # self.coefficients = torch.nn.utils.parametrizations.orthogonal(self.coefficients, use_trivialization=False)
        # print(self.coefficients.parametrizations.weight.original)

    def forward(self, K_train, y_train, K_val):
        sigma = torch.exp(self.sigma_exponent*np.log(10.0))
        n_train = K_train.shape[0] 
        c = torch.linalg.solve(
        self.coefficients(K_train).squeeze(dim = -1) +  # nu = 1, ..., 4 kernels
        sigma * torch.eye(n_train)  # regularization
        , 
        y_train)
        y_val_predictions = self.coefficients(K_val).squeeze(dim = -1) @ c

        return y_val_predictions
