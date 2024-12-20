import torch
import torch.nn as nn
import pickle

### ------------------------------------------------------------------------------------------------------------------
#  Initialize learnable embedding tables
### ------------------------------------------------------------------------------------------------------------------

class EmbeddingTables(nn.Module):
    def __init__(self, vocab_size, n_embd, focus_len):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(focus_len, n_embd)

        # Initialize embeddings
        torch.nn.init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.position_embedding_table.weight, mean=0.0, std=0.02)

        # # Optionally save embedding tables after initialization
        # with open("token_embed.pkl", "wb") as f:
        #     pickle.dump(self.token_embedding_table, f)

        # with open("position_embed.pkl", "wb") as f:
        #     pickle.dump(self.position_embedding_table, f)

    def forward(self, index, device='cpu'):
        # # Ensure token indices are within range
        # index = torch.tensor(index, device=device)
        # assert (index >= 0).all() and (index < self.token_embedding_table.weight.size(0)).all(), \
        #     f"Token indices out of range: {index}"

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(len(index), device = device))

        x = tok_emb + pos_emb
        return x
    
    def update_embeds(self, vocab, new_vocab):
        # Update token embedding table
        with open('token_embed.pkl', 'rb') as f:
            self.token_embedding_table = pickle.load(f)

        old_token_table = self.token_embedding_table.weight.detach().numpy()
        old_token_embedding_table = torch.tensor(old_token_table)

        new_token_embedding_table = old_token_embedding_table.clone()

        for token in new_vocab:
            if token not in vocab:
                random_embedding = torch.empty(1, old_token_embedding_table.size(1)).normal_(mean=0.0, std=0.02)
                new_token_embedding_table = torch.cat([new_token_embedding_table, random_embedding], dim=0)

        nums, dims = new_token_embedding_table.size()
        self.new_token_embedding_table = nn.Embedding(nums, dims)

        with open('token_embed.pkl', 'wb') as f:
            pickle.dump(self.new_token_embedding_table, f)

        # Update position embedding table
        with open('position_embed.pkl', 'rb') as f:
            self.position_embedding_table = pickle.load(f)

        old_position_table = self.position_embedding_table.weight.detach().numpy()
        old_position_embedding_table = torch.tensor(old_position_table)

        new_position_embedding_table = old_position_embedding_table.clone()

        for token in new_vocab:
            if token not in vocab:
                random_embedding = torch.empty(1, old_position_embedding_table.size(1)).normal_(mean=0.0, std=0.02)
                new_position_embedding_table = torch.cat([new_position_embedding_table, random_embedding], dim=0)
        
        nums, dims = new_position_embedding_table.size()
        self.new_position_embedding_table = nn.Embedding(nums, dims)

        with open('position_embed.pkl', 'wb') as f:
            pickle.dump(self.new_position_embedding_table, f)
