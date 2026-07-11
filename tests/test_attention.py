"""Unittest for MultiHeadAttention"""

import pytest
import torch
from DoraGen.attention import Head, MultiHeadAttention

@pytest.fixture
def model_params():
    return {
        "batch_size": 4,
        "context_length": 16,
        "n_embed": 32,
        "n_head": 4,
        "head_size": 8 # n_embed // n_head
    }

@pytest.fixture
def dummy_input(model_params):
    # Tạo tensor đầu vào giả lập (B, T, C)
    torch.manual_seed(42) # Giữ cố định seed để kết quả test nhất quán
    return torch.randn(
        model_params["batch_size"], 
        model_params["context_length"], 
        model_params["n_embed"]
    )

# --- CÁC TEST CASE CHO SINGLE HEAD ---

def test_head_output_shape(model_params, dummy_input):
    """Kiểm tra xem đầu ra của 1 Head có đúng shape (B, T, head_size) không."""
    head = Head(
        head_size=model_params["head_size"],
        n_embd=model_params["n_embed"], # Nhớ sửa typo n_embd -> n_embed trong code chính
        context_length=model_params["context_length"]
    )
    
    out = head(dummy_input)
    
    expected_shape = (
        model_params["batch_size"], 
        model_params["context_length"], 
        model_params["head_size"]
    )
    assert out.shape == expected_shape, f"Kỳ vọng shape {expected_shape}, nhưng nhận được {out.shape}"

def test_head_no_nan_or_inf(model_params, dummy_input):
    """Kiểm tra việc chia scale_factor hoặc softmax có bị lỗi NaN hay Inf không."""
    head = Head(
        head_size=model_params["head_size"],
        n_embd=model_params["n_embed"],
        context_length=model_params["context_length"]
    )
    out = head(dummy_input)
    
    assert not torch.isnan(out).any(), "Output chứa giá trị NaN!"
    assert not torch.isinf(out).any(), "Output chứa giá trị vô cực (Inf)!"

def test_causal_masking_logic(model_params):
    """
    Kiểm tra tính chất Causal Masking: Token ở tương lai KHÔNG ĐƯỢC 
    ảnh hưởng đến kết quả của token trong quá khứ.
    """
    head = Head(
        head_size=model_params["head_size"],
        n_embd=model_params["n_embed"],
        context_length=model_params["context_length"]
    )
    head.eval() # Tắt dropout nếu có

    # Tạo input 1
    x1 = torch.randn(1, model_params["context_length"], model_params["n_embed"])
    
    # Tạo input 2 giống hệt input 1, NHƯNG thay đổi giá trị ở token cuối cùng (tương lai)
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(1, model_params["n_embed"])
    
    out1 = head(x1)
    out2 = head(x2)
    
    # Token đầu tiên (t = 0) phải cho ra kết quả y hệt nhau ở cả 2 input
    # vì masked attention ngăn nó nhìn thấy sự thay đổi ở token cuối cùng
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-6), \
        "Causal masking bị lỗi! Token quá khứ đang bị ảnh hưởng bởi token tương lai."

# --- CÁC TEST CASE CHO MULTI-HEAD ATTENTION ---

def test_multi_head_attention_shape(model_params, dummy_input):
    """Kiểm tra MultiHeadAttention có giữ nguyên kích thước embedding (B, T, C) không."""
    mha = MultiHeadAttention(
        n_head=model_params["n_head"],
        n_embed=model_params["n_embed"],
        context_length=model_params["context_length"]
    )
    
    out = mha(dummy_input)
    assert out.shape == dummy_input.shape, f"Kỳ vọng shape {dummy_input.shape}, nhận được {out.shape}"

def test_backward_pass(model_params, dummy_input):
    """Kiểm tra mô hình có tính được đạo hàm (gradients) để train không bị lỗi."""
    mha = MultiHeadAttention(
        n_head=model_params["n_head"],
        n_embed=model_params["n_embed"],
        context_length=model_params["context_length"]
    )
    
    # Forward pass
    out = mha(dummy_input)
    
    # Tạo một hàm loss giả lập và gọi backward
    loss = out.sum()
    loss.backward()
    
    # Kiểm tra xem các trọng số (weights) đã nhận được gradient chưa
    for name, param in mha.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Layer {name} không nhận được gradient!"
            assert not torch.isnan(param.grad).any(), f"Gradient của {name} bị NaN!"
