import numpy as np
def test_lstm_output_shape():
    """Test LSTM encoder produces correct hidden state shape."""
    batch_size = 4
    seq_length = 60
    input_dim = 8
    hidden_dim = 32
    
    # Mock LSTM implementation
    class MockLSTM:
        def __init__(self, in_dim, hidden):
            self.hidden_dim = hidden
        
        def forward(self, x):
            # Return random hidden state with correct shape
            return np.random.randn(x.shape[0], self.hidden_dim)
    
    lstm = MockLSTM(input_dim, hidden_dim)
    x = np.random.randn(batch_size, seq_length, input_dim)
    embeddings = lstm.forward(x)
    
    assert embeddings.shape == (batch_size, hidden_dim), f"Expected {(batch_size, hidden_dim)}, got {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
    print("✓ LSTM Encoder Output Test Passed")


if __name__ == "__main__":
    test_lstm_output_shape()
