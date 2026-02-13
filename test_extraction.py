import torch
import torch.nn.functional as F
from attention_map_diffusers.modules import joint_attn_call2_0, flux_attn_call2_0, scaled_dot_product_attention
from einops import rearrange

class MockProcessor:
    def __init__(self):
        self.store_attn_map = True

class MockAttention:
    def __init__(self, heads=1):
        self.heads = heads
        self.norm_q = self.norm_k = self.norm_added_q = self.norm_added_k = None
        self.context_pre_only = False
        self.to_out = [lambda x: x, lambda x: x]
    
    # Identity projections to keep values predictable
    def to_q(self, x): return x
    def to_k(self, x): return x
    def to_v(self, x): return x
    def add_q_proj(self, x): return x
    def add_k_proj(self, x): return x
    def add_v_proj(self, x): return x
    def to_add_out(self, x): return x

def test_sd3_joint_extraction():
    print("Testing SD3 Joint Extraction Logic...")
    proc = MockProcessor()
    attn = MockAttention()
    
    # Image tokens: 2, Text tokens: 3
    # SD3 Order: [Image, Text]
    img_len, txt_len = 2, 3
    h, w = 1, 2 # height * width = 2
    
    # Create distinct queries and keys
    # Image query = 1, Text query = 10
    # Image key = 1, Text key = 100
    q_img = torch.ones(1, img_len, 8) * 1.0
    q_txt = torch.ones(1, txt_len, 8) * 10.0
    
    k_img = torch.ones(1, img_len, 8) * 1.0
    k_txt = torch.ones(1, txt_len, 8) * 100.0
    
    # joint_attn_call2_0 will concatenate these internally
    # We pass q_img as hidden_states and q_txt as encoder_hidden_states
    joint_attn_call2_0(proc, attn, q_img, q_txt, height=h, timestep=torch.tensor([0]))
    
    # Reconstruct what the full attention matrix looks like before softmax
    # Q_img * K_img = 1 * 1 = 1
    # Q_img * K_txt = 1 * 100 = 100
    # The softmax will make the 100s dominant in the cross-attention slice
    # and the 1s dominant in the self-attention slice IF THEY WERE SEPARATE.
    # But since they are joint, we just check if the shapes and values align.
    
    print(f"Extracted Cross Map shape: {proc.attn_map.shape}") # Should be (1, 1, 1, 2, 3)
    print(f"Extracted Self Map shape: {proc.self_attn_map.shape}") # Should be (1, 1, 1, 2, 2)
    
    assert proc.attn_map.shape == (1, 1, h, w, txt_len)
    assert proc.self_attn_map.shape == (1, 1, h, w, img_len)
    
    # Verify values: In SD3, Cross is rows[:img] cols[img:]
    # All tokens in proc.attn_map (cross) should have come from the interaction with text keys (100s)
    # All tokens in proc.self_attn_map (self) should have come from interaction with image keys (1s)
    # Since we used identity, the raw scores were Q@K.T.
    # Scores: [1*1, 1*1] for self part, [1*100, 1*100, 1*100] for cross part
    # After softmax, cross should be much larger than self.
    
    cross_mean = proc.attn_map.mean()
    self_mean = proc.self_attn_map.mean()
    
    print(f"Cross mean: {cross_mean:.4f}, Self mean: {self_mean:.4f}")
    assert cross_mean > self_mean, "Extraction might be swapped! Cross should have the high-value text-key interactions."
    print("SD3 Joint Extraction: SUCCESS")

def test_flux_extraction():
    print("Testing Flux Extraction Logic...")
    proc = MockProcessor()
    attn = MockAttention()
    
    # Flux Order: [Text, Image]
    img_len, txt_len = 2, 3
    h, w = 1, 2
    
    q_img = torch.ones(1, img_len, 8) * 1.0
    q_txt = torch.ones(1, txt_len, 8) * 10.0
    
    # Flux concatenates [Text, Image]
    # We call flux_attn_call2_0(proc, attn, hidden_states=q_img, encoder_hidden_states=q_txt, ...)
    flux_attn_call2_0(proc, attn, q_img, q_txt, height=h, timestep=torch.tensor([0]))
    
    # In Flux, Cross is rows[txt:] cols[:txt]
    # Self is rows[txt:] cols[txt:]
    
    print(f"Extracted Cross Map shape: {proc.attn_map.shape}") # (1, 1, 1, 2, 3)
    print(f"Extracted Self Map shape: {proc.self_attn_map.shape}") # (1, 1, 1, 2, 2)
    
    assert proc.attn_map.shape == (1, 1, h, w, txt_len)
    assert proc.self_attn_map.shape == (1, 1, h, w, img_len)
    
    # Again, cross-attention (img-to-txt) uses txt keys (100s), self uses img keys (1s)
    cross_mean = proc.attn_map.mean()
    self_mean = proc.self_attn_map.mean()
    
    print(f"Cross mean: {cross_mean:.4f}, Self mean: {self_mean:.4f}")
    assert cross_mean > self_mean, "Extraction might be swapped!"
    print("Flux Extraction: SUCCESS")

if __name__ == "__main__":
    with torch.no_grad():
        test_sd3_joint_extraction()
        test_flux_extraction()
