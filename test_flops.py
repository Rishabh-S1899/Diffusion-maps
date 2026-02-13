import torch
import torch.nn as nn
from attention_map_diffusers.utils import register_flops_hook, flop_counter
from attention_map_diffusers.modules import joint_attn_call2_0

class MockProcessor:
    def __init__(self):
        self.collect_cross_attn = True
        self.collect_self_attn = True

class MockAttention:
    def __init__(self):
        self.heads = 4
        self.to_q = nn.Linear(16, 16)
        self.to_k = nn.Linear(16, 16)
        self.to_v = nn.Linear(16, 16)
        self.add_q_proj = nn.Linear(16, 16)
        self.add_k_proj = nn.Linear(16, 16)
        self.add_v_proj = nn.Linear(16, 16)
        self.to_out = [nn.Linear(16, 16), nn.Identity()]
        self.norm_q = self.norm_k = self.norm_added_q = self.norm_added_k = None
        self.context_pre_only = False
        self.to_add_out = nn.Identity()

def test_flops():
    print("Testing FLOP Counter...")
    
    # 1. Test Linear Hook
    lin = nn.Linear(10, 20)
    register_flops_hook(lin)
    flop_counter.reset()
    
    x = torch.randn(1, 10)
    lin(x)
    
    # Linear FLOPs = B * L * (2 * Cin - 1) * Cout
    # 1 * 1 * (2 * 10 - 1) * 20 = 19 * 20 = 380
    print(f"Linear FLOPs: {flop_counter.total_flops}")
    assert flop_counter.total_flops == 380
    
    # 2. Test Attention FLOPs
    processor = MockProcessor()
    attn = MockAttention()
    register_flops_hook(attn.to_q) # register on one component
    
    flop_counter.reset()
    
    # B=1, H=4, Lq=8, Lk=4, D=4 (16/4 heads)
    q = torch.randn(1, 8, 16)
    k = torch.randn(1, 4, 16)
    
    # We call the processor directly to test the manual count
    joint_attn_call2_0(processor, attn, q, k, height=1, timestep=torch.tensor([0]))
    
    print(f"Total FLOPs after Attention: {flop_counter.total_flops}")
    assert flop_counter.total_flops > 0
    assert "attention_q_k" in flop_counter.layer_flops
    
    flop_counter.print_summary()
    print("FLOP Verification SUCCESSFUL!")

if __name__ == "__main__":
    test_flops()
