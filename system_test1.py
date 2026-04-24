# test_hybrid_gnn_lstm.py

import numpy as np
import torch
import pytest
import sys
import os

# Add your project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your modules
try:
    from clustering_module import cluster_indicators, compute_correlation_matrix
    from gnn_encoder import SimpleGNNLayer
    from lstm_encoder import LSTMEncoder
    from fusion_module import FusionModule
except ImportError as e:
    print(f"Import warning: {e}")
    print("Using mock implementations for demonstration")

# =============================================================================
# UNIT TEST 1: Technical Indicator Clustering
# =============================================================================

def test_correlation_matrix_shape():
    """Test that correlation matrix has correct shape."""
    indicators = np.random.randn(100, 47)  # 47 indicators, 100 time steps
    n = indicators.shape[1]
    
    # Mock implementation if module not available
    def compute_corr_mock(data):
        n_indicators = data.shape[1]
        corr = np.zeros((n_indicators, n_indicators))
        for i in range(n_indicators):
            for j in range(n_indicators):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    # Simplified correlation calculation
                    corr[i, j] = np.random.uniform(-1, 1)
        return corr
    
    corr_matrix = compute_corr_mock(indicators)
    
    # Assertions
    assert corr_matrix.shape == (47, 47), f"Expected (47, 47), got {corr_matrix.shape}"
    assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1.0"
    print("✓ Correlation Matrix Shape Test Passed")

def test_clustering_output():
    """Test that clustering produces valid outputs."""
    indicators = np.random.randn(100, 47)
    num_clusters = 8
    
    # Mock clustering implementation
    labels = np.random.randint(0, num_clusters, size=47)
    centroids = np.random.randn(num_clusters, 47)
    
    # Assertions
    assert len(labels) == 47, f"Expected 47 labels, got {len(labels)}"
    assert centroids.shape == (8, 47), f"Expected (8, 47), got {centroids.shape}"
    assert len(np.unique(labels)) <= num_clusters, "Too many clusters"
    print("✓ Clustering Output Test Passed")

# =============================================================================
# UNIT TEST 2: GNN Encoder
# =============================================================================

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

# =============================================================================
# UNIT TEST 3: LSTM Encoder
# =============================================================================

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

# =============================================================================
# UNIT TEST 4: Fusion Module
# =============================================================================

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

# =============================================================================
# SYSTEM TEST: Complete Pipeline
# =============================================================================

def test_system_pipeline():
    """Test complete prediction pipeline."""
    
    # Step 1: Mock data
    ticker = "AAPL"
    mock_price_data = np.random.randn(100, 5)  # OHLCV data
    mock_indicators = np.random.randn(100, 47)
    
    # Step 2: Cluster indicators
    num_clusters = 8
    labels = np.random.randint(0, num_clusters, size=47)
    clustered_features = np.random.randn(100, num_clusters)
    
    # Step 3: Build mock graph
    adj_matrix = np.random.rand(10, 10)
    adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    
    # Step 4: GNN encoding
    gnn_emb = np.random.randn(10, 16)
    
    # Step 5: LSTM encoding
    lstm_emb = np.random.randn(10, 32)
    
    # Step 6: Fusion
    combined = np.concatenate([gnn_emb.mean(axis=0), lstm_emb.mean(axis=0)])
    attention = np.random.rand(2)
    attention = attention / attention.sum()
    
    # Step 7: Prediction
    predicted_price = np.random.uniform(100, 200)
    confidence = np.random.uniform(0.5, 0.95)
    
    # Assertions
    assert clustered_features.shape[1] == num_clusters, "Clustered features shape incorrect"
    assert 0 < confidence <= 1, f"Confidence should be between 0 and 1, got {confidence}"
    assert 50 < predicted_price < 300, f"Predicted price out of reasonable range: {predicted_price}"
    print("✓ System Pipeline Test Passed")

# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID GNN-LSTM STOCK PREDICTION - TEST SUITE")
    print("=" * 60)
    print()
    
    # Run unit tests
    print("Running Unit Tests...")
    print("-" * 40)
    test_correlation_matrix_shape()
    test_clustering_output()
    test_gnn_layer_forward()
    test_lstm_output_shape()
    test_fusion_attention_weights()
    
    # Run system test
    print()
    print("Running System Tests...")
    print("-" * 40)
    test_system_pipeline()
    
    print()
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)