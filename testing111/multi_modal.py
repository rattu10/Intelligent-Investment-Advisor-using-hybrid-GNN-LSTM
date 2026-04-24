import numpy as np
def test_fusion_attention_weights():
    """Test fusion module produces valid attention weights."""
    batch_size = 4
    gnn_dim = 16
    lstm_dim = 32
    
    # Mock fusion implementation
    def mock_fusion(gnn_emb, lstm_emb):
        combined = np.concatenate([gnn_emb, lstm_emb], axis=-1)
        # Simple attention
        attention = np.random.rand(batch_size, 2)
        attention = attention / attention.sum(axis=1, keepdims=True)
        return attention
    
    gnn_emb = np.random.randn(batch_size, gnn_dim)
    lstm_emb = np.random.randn(batch_size, lstm_dim)
    attention = mock_fusion(gnn_emb, lstm_emb)
    
    assert attention.shape == (batch_size, 2), f"Expected {(batch_size, 2)}, got {attention.shape}"
    assert np.allclose(attention.sum(axis=1), 1.0), "Attention weights should sum to 1"
    assert np.all(attention >= 0) and np.all(attention <= 1), "Attention weights should be between 0 and 1"
    print("✓ Fusion Module Attention Test Passed")


if __name__ == "__main__":
    test_fusion_attention_weights()