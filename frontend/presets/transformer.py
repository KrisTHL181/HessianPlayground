class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(dim)
        self.ln2 = torch.nn.LayerNorm(dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, ff_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ff_dim, dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class TransformerMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_size, hidden_sizes[0])
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 1, hidden_sizes[0]) * 0.02)

        blocks = []
        for i, h in enumerate(hidden_sizes):
            blocks.append(TransformerBlock(h, num_heads=4, ff_dim=h * 4, dropout=0.1))
        self.blocks = torch.nn.ModuleList(blocks)
        self.output = torch.nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1).unsqueeze(1)  # (B, 1, D)
        x = self.input_proj(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = x.squeeze(1)  # (B, D)
        return self.output(x)
