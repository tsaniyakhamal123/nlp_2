import numpy as np

class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.E = np.random.randn(vocab_size, d_model) * 0.01

    def __call__(self, tokens):
        return self.E[tokens]

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = pos / np.power(10000, (2 * (i // 2)) / d_model)
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angle_rates[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rates[:, 1::2])
        self.pe = pe

    def __call__(self, x):
        return x + self.pe[:x.shape[0]]

class RotaryPositionalEncoding:
    def __init__(self, d_model, max_len=2048):
        assert d_model % 2 == 0, "d_model harus genap untuk RoPE"
        inv_freq = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))
        pos = np.arange(0, max_len)
        sinusoid_inp = np.outer(pos, inv_freq)
        self.sin = np.sin(sinusoid_inp)
        self.cos = np.cos(sinusoid_inp)

    def apply(self, x):
        # x shape: (batch, heads, seq_len, d_k)
        seq_len = x.shape[2]
        d_k = x.shape[3]

        assert d_k % 2 == 0, "d_k harus genap untuk RoPE"

        sin = self.sin[:seq_len, :d_k // 2].reshape(1, 1, seq_len, d_k // 2)
        cos = self.cos[:seq_len, :d_k // 2].reshape(1, 1, seq_len, d_k // 2)


        x1 = x[..., ::2]  # genap
        x2 = x[..., 1::2]  # ganjil

        x_rotated = np.concatenate([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], axis=-1)

        return x_rotated
