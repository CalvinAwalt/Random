# Optional: Neural network filter (PyTorch)
class UsefulnessPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder(d_model=128, nhead=8)
        
    def forward(self, equation_str):
        # Convert equation to token embeddings
        tokens = tokenize(equation_str)
        return self.encoder(tokens)  # Output: [0-1 usefulness score]