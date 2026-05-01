class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(200, 10)
        self.labels = torch.randint(0, 3, (200,))
    def __len__(self): return 200
    def __getitem__(self, i): return self.data[i], self.labels[i]