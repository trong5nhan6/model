import torch
import torch.nn as nn
from einops import repeat, pack, unpack  # Cần cài einops: pip install einops

# =========================
# Giả lập class chứa hàm
# =========================
class TestEmbed:
    def __init__(self, num_tokens, dim, n_registers):
        self.input_embed = nn.Embedding(num_tokens, dim)
        # register tokens là một tensor learnable (có thể random)
        self.register_tokens = nn.Parameter(torch.randn(n_registers, dim))

    def embed_inputs_with_registers(self, seq):
        batch = seq.shape[0]

        inputs = self.input_embed(seq) # b n d (2 4 4)

        # lặp registers cho mỗi batch
        registers = repeat(self.register_tokens, 'n d -> b n d', b=batch) # b n' d (2 3 4)

        # gộp registers + inputs
        inputs, packed_shape = pack([registers, inputs], 'b * d') # (2 3+4 4)

        return inputs, packed_shape

# =========================
# Test hàm
# =========================
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    num_tokens = 10
    dim = 4
    n_registers = 3

    # khởi tạo object
    tester = TestEmbed(num_tokens=num_tokens, dim=dim, n_registers=n_registers)

    # tạo batch sequence giả lập
    seq = torch.randint(0, num_tokens, (batch_size, seq_len))
    print("Input sequence:\n", seq)

    # gọi hàm
    packed_inputs, packed_shape = tester.embed_inputs_with_registers(seq)
    print("\nPacked inputs shape:", packed_inputs.shape)
    print("Packed inputs:\n", packed_inputs)
    print("Packed shape:", packed_shape)

    # giải nén trở lại
    unpacked = unpack(packed_inputs, packed_shape, 'b * d')
    print("\nUnpacked tensors (registers + inputs):")
    for t in unpacked:
        print(t.shape)
