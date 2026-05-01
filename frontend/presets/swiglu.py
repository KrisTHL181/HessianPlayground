class SwiGLU(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_gate = torch.nn.Linear(d_in, d_out)
        self.w_proj = torch.nn.Linear(d_in, d_out)
        self.w_out = torch.nn.Linear(d_out, d_out)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.w_gate(x))
        proj = self.w_proj(x)
        return self.w_out(gate * proj)


class SwiGLUMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev = input_size
        for i, h in enumerate(hidden_sizes):
            layers.append((f'swiGLU{i}', SwiGLU(prev, h)))
            prev = h
        layers.append(('output', torch.nn.Linear(prev, output_size)))
        self.net = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
