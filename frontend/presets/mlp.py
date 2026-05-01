class SimpleMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev = input_size
        for i, h in enumerate(hidden_sizes):
            layers.append((f'fc{i}', torch.nn.Linear(prev, h)))
            layers.append((f'relu{i}', torch.nn.ReLU()))
            prev = h
        layers.append(('output', torch.nn.Linear(prev, output_size)))
        self.net = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
