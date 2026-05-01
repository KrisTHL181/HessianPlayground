class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, out_dim)
        self.bn1 = torch.nn.BatchNorm1d(out_dim)
        self.fc2 = torch.nn.Linear(out_dim, out_dim)
        self.bn2 = torch.nn.BatchNorm1d(out_dim)
        self.skip = torch.nn.Linear(in_dim, out_dim) if in_dim != out_dim else torch.nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return torch.nn.functional.relu(out + residual)


class ResNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev = input_size
        for i, h in enumerate(hidden_sizes):
            layers.append((f'resblock{i}', ResidualBlock(prev, h)))
            prev = h
        layers.append(('output', torch.nn.Linear(prev, output_size)))
        self.net = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
