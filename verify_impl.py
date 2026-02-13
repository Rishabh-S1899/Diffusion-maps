import torch
import torch.nn.functional as F
from attention_map_diffusers.modules import joint_attn_call2_0, scaled_dot_product_attention
from attention_map_diffusers.utils import hook_function, attn_maps
import json
import os

class MockProcessor:
    def __init__(self):
        self.store_attn_map = True

class MockAttention:
    def __init__(self):
        self.heads = 4
        self.norm_q = None
        self.norm_k = None
        self.norm_added_q = None
        self.norm_added_k = None
        self.context_pre_only = False
        self.to_out = [lambda x: x, lambda x: x]
        self.spatial_norm = None
        self.group_norm = None
        self.norm_cross = False
        self.residual_connection = True
        self.rescale_output_factor = 1.0
        
    def to_q(self, x): return x
    def to_k(self, x): return x
    def to_v(self, x): return x
    def add_q_proj(self, x): return x
    def add_k_proj(self, x): return x
    def add_v_proj(self, x): return x
    def to_add_out(self, x): return x
    def prepare_attention_mask(self, mask, seq_len, batch_size): return mask

def verify():
    print("Starting verification of self-attention implementation...")
    
    processor = MockProcessor()
    attn = MockAttention()
    
    # Mock data: Batch=1, Heads=4, Image_Tokens=16 (4x4), Text_Tokens=4, Head_Dim=8
    height, width = 4, 4
    batch_size = 1
    head_dim = 8
    
    class MockModule:
        def __init__(self, p): self.processor = p
    
    module = MockModule(processor)
    hook = hook_function("test_layer")

    # Simulate Timestep 1
    print("\nSimulating Timestep 1...")
    hidden_states = torch.randn(batch_size, 16, attn.heads * head_dim)
    encoder_hidden_states = torch.randn(batch_size, 4, attn.heads * head_dim)
    timestep = torch.tensor([1000.0])
    
    joint_attn_call2_0(processor, attn, hidden_states, encoder_hidden_states, 
                       height=height, timestep=timestep)
    
    print(f"Timestep 1 - Cross Similarity: {processor.similarity}")
    print(f"Timestep 1 - Self Similarity: {processor.self_similarity}")
    
    # Trigger hook for timestep 1
    hook(module, None, None)
    
    # Simulate Timestep 2 (slightly changed data)
    print("\nSimulating Timestep 2...")
    hidden_states_2 = hidden_states + 0.1 * torch.randn_like(hidden_states)
    timestep_2 = torch.tensor([980.0])
    
    joint_attn_call2_0(processor, attn, hidden_states_2, encoder_hidden_states, 
                       height=height, timestep=timestep_2)
    
    print(f"Timestep 2 - Cross Similarity: {processor.similarity:.4f}")
    print(f"Timestep 2 - Self Similarity: {processor.self_similarity:.4f}")
    
    # Trigger hook for timestep 2
    hook(module, None, None)
    
    # Test Stats Saving
    print("\nTesting Stats Saving...")
    from attention_map_diffusers.utils import save_attention_stats
    save_attention_stats(attn_maps, base_dir='test_verify_output')
    
    with open('test_verify_output/statistics.json', 'r') as f:
        stats = json.load(f)
        
        # JSON keys from dictionary with int keys become strings
        assert "1000" in stats, "Missing timestep 1000 in JSON"
        assert "980" in stats, "Missing timestep 980 in JSON"
        
        print("\nGenerated JSON sample (Timestep 980):")
        print(json.dumps(stats["980"]["test_layer"], indent=2))
        
        layer_stats = stats["980"]["test_layer"]
        assert 'similarity' in layer_stats
        assert 'entropy' in layer_stats
        assert 'self_similarity' in layer_stats
        assert 'self_entropy' in layer_stats
        assert layer_stats['similarity'] > 0, "Similarity should be positive"
        assert layer_stats['self_similarity'] > 0, "Self-similarity should be positive"

    print("\nVerification SUCCESSFUL!")

if __name__ == "__main__":
    verify()
