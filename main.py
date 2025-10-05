import numpy as np
from embedding import TokenEmbedding, PositionalEncoding, RotaryPositionalEncoding
from attention import MultiHeadAttention, causal_mask, visualize_attention
from ffn import FeedForward, layer_norm
from attention import softmax

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, use_rope=False):
        self.attn = MultiHeadAttention(d_model, num_heads, use_rope)
        self.ffn = FeedForward(d_model, d_ff)

    def __call__(self, x, mask=None, return_attn=False, rope=None):
        attn_out = self.attn(layer_norm(x), mask, return_attn, rope)
        x = x + attn_out
        ffn_out = self.ffn(layer_norm(x))
        x = x + ffn_out
        return x

class GPT:
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=256, num_layers=2, max_len=50, use_rope=False):
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len)
        self.rope = RotaryPositionalEncoding(d_model, max_len) if use_rope else None
        self.layers = [TransformerBlock(d_model, num_heads, d_ff, use_rope) for _ in range(num_layers)]
        self.W_vocab = self.embedding.E.T  # weight tying

    def __call__(self, tokens, return_attn=False):
        x = self.embedding(tokens)
        x = self.positional(x)
        x = np.expand_dims(x, 0)
        mask = causal_mask(x.shape[1])[None, None, :, :]

        attn_weights_all = []
        for layer in self.layers:
            x = layer(x, mask, return_attn, self.rope)
            if return_attn:
                attn_weights_all.append(layer.attn.attn_weights)

        logits = x @ self.W_vocab
        probs = softmax(logits, axis=-1)

        if return_attn:
            return logits, probs, attn_weights_all
        return logits, probs

# === RUN TEST ===
if __name__ == "__main__":
    vocab_size = 20
    tokens = np.array([2, 5, 7, 10])
    model = GPT(vocab_size=vocab_size, d_model=32, num_heads=4, d_ff=128, num_layers=2, max_len=10, use_rope=True)
    logits, probs, attn_maps = model(tokens, return_attn=True)

    print("✅ Logits shape:", logits.shape)
    print("✅ Probabilitas token terakhir:", probs[0, -1])
    print("✅ Jumlah probabilitas:", probs[0, -1].sum())

    visualize_attention(attn_maps, layer=0, head=0)
