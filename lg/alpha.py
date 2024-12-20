import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer, trainers, pre_tokenizers, decoders

from arch import GetTopTokens


# from arch import Block, MultiHeadAttention, FeedForward, Head, LGGPT

class AlphaZeroTrainer:
    def __init__(self, model, embeds, tokenizer, single_head, vocab, lr):
        self.model = model
        self.embeds = embeds
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=0.0001)
        self.single_head = single_head

    def mcts_search(self, sentence, n_sims):
        tree = {}  # Monte Carlo Tree: stores N(s, a), Q(s, a), P(s, a)

        for _ in range(n_sims):
            # Step 1: Selection
            path, node = self.select_node(tree, sentence)

            tids = self.tokenizer.encode(" ".join(map(str, node))).ids
            s_tensor = torch.tensor(tids).unsqueeze(0)

            # Step 2: Expansion
            self.expand_node(tree, node)

            # Step 3: Rollout (evaluate using network)
            logits, _ = self.model.generate_forward(self.embeds, s_tensor)
            policy_probs = F.softmax(logits, dim=-1)
            value_estimate = torch.mean(logits)  # Use logits mean or a value head

            # Step 4: Backpropagation
            self.backpropagate(tree, path, value_estimate)

        return tree

    def select_node(self, tree, top_toks):
        path = []
        current_node = tuple(top_toks)

        while current_node in tree and tree[current_node]["children"]:
            children = tree[current_node]["children"]
            total_visits = sum(tree[child]["N"] for child in children)

            if not children:  # Handle empty children case
                print(f"No children for node {current_node}")
                break

            # Select the child with the highest UCB score
            best_child = max(
                children,
                key=lambda child: (
                    tree[child]["Q"] +
                    np.sqrt(total_visits) * tree[child]["P"] / (1 + tree[child]["N"])
                )
            )
            path.append((current_node, best_child))
            current_node = best_child

        return path, current_node



    def expand_node(self, tree, node):
        if node in tree and "children" in tree[node] and tree[node]["children"]:
            return

        # If node contains token IDs, use them directly
        if isinstance(node, list) and all(isinstance(x, int) for x in node):
            token_ids = node  # Use node directly if it contains token IDs
        else:
            # Convert node tokens to a string and then tokenize
            token_ids = self.tokenizer.encode(" ".join(map(str, node))).ids

        sentence_tensor = torch.tensor(token_ids).unsqueeze(0)

        logits, _ = self.model.generate_forward(self.embeds, sentence_tensor)
        # Ensure policy_probs is a NumPy array
        policy_probs = F.softmax(logits, dim=-1).detach().cpu().numpy()

        # Flatten policy_probs to 1D if needed
        policy_probs = policy_probs.flatten()

        # Debugging print (optional)
        print(f"policy_probs: {policy_probs}, shape: {policy_probs.shape}")

        # Add children for all possible tokens
        tree[node] = {"N": 0, "Q": 0, "P": 0, "children": []}
        for token_id in range(policy_probs.shape[0]):
            prob = policy_probs[token_id]
            if prob > 1e-5:
                child = tuple(list(node) + [token_id])
                tree[child] = {"N": 0, "Q": 0, "P": prob, "children": []}
                tree[node]["children"].append(child)  # Append to parent's children list


    def backpropagate(self, tree, path, value):
        for parent, child in reversed(path):
            tree[child]["N"] += 1
            tree[child]["Q"] += (value - tree[child]["Q"]) / tree[child]["N"]

    def extract_targets(self, tree):
        root_node = list(tree.keys())[0]  # Assume root is the first key in the tree
        children = tree[root_node]["children"]

        # Policy target: normalized visit counts
        visit_counts = np.array([tree[child]["N"] for child in children])
        policy_target = visit_counts / visit_counts.sum()

        # Value target: Q-value of the root
        value_target = np.mean([tree[child]["Q"].detach().numpy() if torch.is_tensor(tree[child]["Q"]) else tree[child]["Q"] for child in children])


        return torch.tensor(policy_target, dtype=torch.float32), torch.tensor(value_target, dtype=torch.float32)
    
    def train_step(self, sentence, n_sims, focus_len:int = 256):
        # Step 1: Tokenize the input sentence
        sentence_tokens = self.tokenizer.encode(sentence).ids  # Convert to token IDs
        print("sentence tokens: ", sentence_tokens)

        ########################################################################################################
        ########################################################################################################
        bos_token = torch.tensor([2])  # <bos>
        eos_token = torch.tensor([3])  # <eos>

        data = torch.tensor(sentence_tokens)
        data = torch.cat([bos_token, data, eos_token])

        print("data: ", data)
        print("data shape: ", data.shape)

        if len(data) > focus_len:     # Truncate long entries
            index = data[-focus_len:]
        elif len(data) < focus_len:   # Pad short entries
            empty_add = torch.tensor([1] * (focus_len - len(data)))  # <pad> token is 1
            index = torch.cat([empty_add, data])
        else:
            index = data

        print("index size: ", index.shape)

        x = self.embeds.forward(index, device='cpu')
        print("x shape: ", x.shape)

        attended = self.single_head.forward(x)
        print("attended shape: ", attended.shape)
        print("attended: ", attended)

        ##
        ### Extract the top 3 priority tokens
        ##

        new_top_tokens, new_top_tokenids = GetTopTokens(attended, index, self.vocab, tokenizer=self.tokenizer, k=4)
        print("top tokens shape: ", new_top_tokenids.shape)

        ########################################################################################################
        ########################################################################################################

        # Step 2: Perform MCTS to get policy and value targets
        tree = self.mcts_search(new_top_tokenids, n_sims=n_sims)
        policy_target, value_target = self.extract_targets(tree)

        # Step 3: Convert sentence to indices
        sentence_tensor = torch.tensor(sentence_tokens, dtype=torch.long).unsqueeze(0)

        # Step 4: Forward pass through the model
        logits, loss = self.model.train_forward(self.embeds, sentence_tensor, value_target)

        # Step 5: Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
