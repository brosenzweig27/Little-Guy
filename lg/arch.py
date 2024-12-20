import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer, trainers, pre_tokenizers, decoders

### ------------------------------------------------------------------------------------------------------------------
#  Define transformer blocks
### ------------------------------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, focus_len):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, focus_len)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


### ------------------------------------------------------------------------------------------------------------------
#  Define multi-head attention
### ------------------------------------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout, focus_len):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout, focus_len) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


### ------------------------------------------------------------------------------------------------------------------
#  Define feed forward sequence
### ------------------------------------------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

### ------------------------------------------------------------------------------------------------------------------
#  Define individual attention head
### ------------------------------------------------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size, n_embd, dropout, focus_len):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(focus_len, focus_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[1]  # Sequence length
        if self.tril.shape[0] < T:  # If tril is too small, regenerate it
            self.tril = torch.tril(torch.ones(T, T, device=x.device))
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Apply mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out
    
    
### ------------------------------------------------------------------------------------------------------------------
#  Initialize Dual-GPT Architecture
### ------------------------------------------------------------------------------------------------------------------
class LGGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, dropout, focus_len):
        super().__init__()

        ##
        ### Create block sequence 1 with distinct blocks
        ##
        self.block_11 = Block(n_embd, n_head=n_head, dropout=dropout, focus_len=focus_len)
        self.block_12 = Block(n_embd, n_head=n_head, dropout=dropout, focus_len=focus_len)
        self.block_13 = Block(n_embd, n_head=n_head, dropout=dropout, focus_len=focus_len)

        self.blocks_1 = nn.Sequential(self.block_11, self.block_12, self.block_13)

        self.seq1_blocks = [self.block_11, self.block_12, self.block_13]

        ##
        ### Create block sequence 2 with distinct blocks
        ##
        self.block_21 = Block(n_embd, n_head=n_head, dropout=dropout, focus_len=focus_len)
        self.block_22 = Block(n_embd, n_head=n_head, dropout=dropout, focus_len=focus_len)
        self.block_23 = Block(n_embd, n_head=n_head, dropout=dropout, focus_len=focus_len)

        self.blocks_2 = nn.Sequential(self.block_21, self.block_22, self.block_23)

        self.seq2_blocks = [self.block_21, self.block_22, self.block_23]

        ##
        ### Create layer norm and linear layer
        ##
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    ##
    ### When vocab changes, add a new linear layer of correct size
    ##
    def update_linlayer(self, old_vocab_size, new_vocab_size):
        new_lm_head = nn.Linear(old_vocab_size, new_vocab_size)
        torch.nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.02)
        if new_lm_head.bias is not None:
            torch.nn.init.zeros_(new_lm_head.bias)
        
        if isinstance(self.lm_head, nn.Linear):
            self.lm_head = nn.Sequential(self.lm_head, new_lm_head)
        else: self.lm_head = nn.Sequential(*self.lm_heah, new_lm_head)

    ##
    ### Train both major pathways
    ##
    def train_forward(self, embeds, index, targets=None):

        x = embeds.forward(index)
        
        x1 = self.blocks_1(x)
        # x2 = self.blocks_2(x)
        x1 = self.ln_f(x1)
        # x2 = self.ln_f(x2)
        logits1 = self.lm_head(x1)
        # logits2 = self.lm_head(x2)

        if targets is None:
            loss = None
        else:
            B1, T1, C1 = logits1.shape
            logits1 = logits1.view(B1*T1, C1)
            targets = targets.view(B1*T1)
            loss1 = F.cross_entropy(logits1, targets)

            # B2, T2, C2 = logits2.shape
            # logits2 = logits2.view(B2*T2, C2)
            # targets = targets.view(B2*T2)
            # loss2 = F.cross_entropy(logits2, targets)

        return logits1, loss1
    
    ##
    ### Forward step for generation
    ##
    def generate_forward(self, embeds, index, targets=None):

        x = embeds.forward(index)

        x = self.blocks_1(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    ##
    ### Select transformer sequence and generate
    ##
    def generate(self, embeds, index, path, max_new_tokens):
        if path == None:
            return None
        else:
            block_path = [self.seq2_blocks[i] if p == 2 else self.seq1_blocks[i] for i, p in enumerate(path)]
            self.curr_blocks = nn.Sequential(*block_path)

        for _ in range(max_new_tokens):
            logits, loss = self.generate_forward(embeds, index, None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
            index = index[:, 1:]
        return index
    
    
### ------------------------------------------------------------------------------------------------------------------
#  Initialize Single-Head Attention for Learning
### ------------------------------------------------------------------------------------------------------------------

class SingleHead(nn.Module):
    def __init__(self, n_embd, head_size, dropout, focus_len):
        super().__init__()
        
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.proj = nn.Linear(head_size, n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(focus_len, focus_len)))

        self.dropout = nn.Dropout(dropout)

        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)           # [T, head_size]
        out = wei @ v               # [T, head_size]
        out = self.proj(out)        # [T, n_embd]

        x = self.ln1(x + out)
        y = self.ffwd(x)
        out = self.ln2(x + y)

        return out
    

### ------------------------------------------------------------------------------------------------------------------
#  Step 7 (L)
### ------------------------------------------------------------------------------------------------------------------

def GetTopTokens(attended, index, vocab, tokenizer, k:int=3):
    avgs = torch.mean(torch.abs(attended), dim=1)
    top_indices = torch.topk(torch.abs(avgs), k=k, dim=0)[1]

    # Create a new tensor with 1s at the top k indices and 0s elsewhere
    binary_tensor = torch.zeros_like(avgs)
    binary_tensor.scatter_(0, top_indices, 1)
    
    top_tokens = binary_tensor.clone().detach().bool()

    decoded_tokens = [tokenizer.decode([token_id]) for token_id in index.tolist()]

    selected_tokens = [token for token, is_top in zip(decoded_tokens, top_tokens) if is_top]
    selected_token_ids = [token_id for token_id, is_top in zip(index.tolist(), top_tokens) if is_top]

    return selected_tokens, selected_token_ids