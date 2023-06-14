import os
import math
import copy
import torch
import numpy as np
import transformers
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Use Apple Silicon
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# Use NVIDIA GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Your using "{device}" as your training device')


class Embedding(nn.Module):
    """The Embedding Layer of Transformer architecture"""
    def __init__(self, vocab_size, embed_dim=512):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: input vector, usually has the size of (batch_size, sequence_len)
        Returns:
            output: embedding vector, usually has the size of (batch_size, sequence_len, embed_dim)
        """
        output = self.embedding(x)
        
        return output


# TODO: Add function descriptions
class MultiHeadAttention(nn.Module):
    """Implementation of MultiHeadAttention mechanism"""
    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model -> int: The dimension per embedding vector
            num_heads -> int: The number of heads of this self-attention
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "The embedding dimension should be divsible by the number of heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.reveal = reveal

        # Initial weights for Key, Query, Value, and Output
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    # TODO: Fix description
    def scaled_dot_product_attention(self, K: torch.Tensor, Q: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Desciption:
            This is the implementation of Scaled Dot-Product Attention

                    Attention(Q, K, V) = softmax(QK_t / √d_k)V
        Args:
            K: Tensor, Key of self-attention, shape ``[batch_size, seq_len, embed_dim]``
            Q: Tensor, Query of self-attention, shape ``[batch_size, seq_len, embed_dim]``
            V: Tensor, Value of self-attention, shape ``[batch_size, seq_len, embed_dim]``
            mask: Tensor, mask for the Transformer decoder, shape
        Returns:
            output: Tensor, the result of Scaled Dot-Product Attention, shape ``[batch_size, seq_len, embed_dim]``
        """
        # QK_t / √d_model, the scaled dot-products value
        # The dim of both Q and V is 32x8x10x64, the dim of V_t will be 32x8x64x10
        # So the dim of attention_score will be 32x8x10x10
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Apply mask
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        
        # Softmax(QK_t / √d_model)
        attention_probs = torch.softmax(attention_score, dim=-1)
        
        # Softmax(QK_t / √d_model) x V
        # attention_score(32x8x10x10) x V(32x8x10x64) -> output dim will be 32x8x10x64
        # Which input dim == output dim
        output = torch.matmul(attention_probs, V)
        
        return output

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Split the embeddings into num_heads channels.

        Args:
            x: Tensor, shape ``[batch_size, seq_len, embed_dim]``
        Returns:
            x: Tensor, shape ``[batch_size, num_heads, seq_len, embed_dim]``
        """
        batch_size, sequence_len, d_model = x.size()
        
        return x.view(batch_size, sequence_len, self.num_heads, self.d_k).transpose(1, 2)

    def fuse_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Desciption:
            Fuse back the splited heads in to one Tensor
        
        Args:
            x: Tensor
        """
        batch_size, _, sequence_len, d_k = x.size()
        
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_len, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            Q: Tensor, Query of the self-attention, shape ``[batch_size, seq_len, embed_dim]``
            K: Tensor, Key of the self-attention, shape ``[batch_size, seq_len, embed_dim]``
            V: Tensor, Value of the self-attention, shape ``[batch_size, seq_len, embed_dim]``
            mask: Tensor, The mask needed for Transformer decoder
        Returns:
            output: Tensor, The output of the MultiHeadAttention mechanism, shape ``[batch_size, seq_len, embed_dim]``
        """
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_O(self.fuse_heads(attention))

        return output