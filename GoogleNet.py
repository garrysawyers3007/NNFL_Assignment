class GoogleNet(nn.Module):
    def __init__(self, original_model):
        super(GoogleNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x