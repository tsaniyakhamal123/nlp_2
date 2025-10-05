import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)

def scaled_dot_product_attention(Q, K, V, mask=None, return_weights=False):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    if mask is not None:
        scores += mask
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return (output, weights) if return_weights else output

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, use_rope=False):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01
        self.Wo = np.random.randn(d_model, d_model) * 0.01
        self.attn_weights = None

    def __call__(self, x, mask=None, return_attn=False, rope=None):
        batch, seq_len, d_model = x.shape

        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        if self.use_rope and rope is not None:
            Q = rope.apply(Q)
            K = rope.apply(K)

        if return_attn:
            out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask, return_weights=True)
        else:
            out = scaled_dot_product_attention(Q, K, V, mask)

        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        return out @ self.Wo

def visualize_attention(attn_weights, layer=0, head=0):
    weights = attn_weights[layer][0, head]
    plt.figure(figsize=(6, 5))
    plt.title(f"Layer {layer}, Head {head}")
    plt.imshow(weights, cmap='viridis')
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.colorbar()
    plt.show()
