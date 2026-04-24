import numpy as np
def test_gnn_layer_forward():
    """Test GNN layer produces correct output shape."""
    num_nodes = 10
    input_dim = 8
    hidden_dim = 16
    batch_size = 4
    
    # Create mock GNN layer
    class MockGNNLayer:
        def __init__(self, in_dim, out_dim):
            self.linear = lambda x: x @ np.random.randn(in_dim, out_dim)
        
        def forward(self, x, adj):
            x = adj @ x
            x = self.linear(x)
            return np.maximum(0, x)  # ReLU
    
    gnn = MockGNNLayer(input_dim, hidden_dim)
    x = np.random.randn(num_nodes, input_dim)
    adj = np.random.randn(num_nodes, num_nodes)
    adj = np.exp(adj) / np.exp(adj).sum(axis=1, keepdims=True)  # Softmax
    
    output = gnn.forward(x, adj)
    
    assert output.shape == (num_nodes, hidden_dim), f"Expected {(num_nodes, hidden_dim)}, got {output.shape}"
    assert not np.isnan(output).any(), "Output contains NaN values"
    print("✓ GNN Layer Forward Test Passed")


if __name__ == "__main__":
    test_gnn_layer_forward()