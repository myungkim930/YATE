# PyTorch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_prob: float,
        num_layers: int,
    ):
        super(SimpleMLP, self).__init__()

        self.mlp_initial = nn.Linear(input_dim, hidden_dim)

        self.mlp_block = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.Sequential(*[self.mlp_block for _ in range(num_layers)])

        self.classifer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = self.mlp_initial(X)
        X = self.layers(X)
        X = self.classifer(X)
        return X
