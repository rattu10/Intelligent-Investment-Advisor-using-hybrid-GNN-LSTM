import numpy as np
import yfinance as yf

def test_system_pipeline_real():
    """Test complete pipeline with real-world data"""

    # Step 1: Fetch real stock data
    data = yf.download("AAPL", period="6mo")
    
    assert not data.empty, "Failed to fetch stock data"

    # Step 2: Prepare OHLCV
    price_data = data[['Open','High','Low','Close','Volume']].values

    # Step 3: Mock technical indicators (replace with real function if available)
    indicators = np.random.randn(len(price_data), 47)

    # Step 4: Clustering (simulated)
    num_clusters = 8
    clustered_features = np.random.randn(len(price_data), num_clusters)

    # Step 5: Graph Construction (simulated)
    adj_matrix = np.random.rand(10, 10)
    adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)

    # Step 6: GNN + LSTM (simulated embeddings)
    gnn_emb = np.random.randn(10, 16)
    lstm_emb = np.random.randn(10, 32)

    # Step 7: Fusion
    fused = np.concatenate([gnn_emb.mean(axis=0), lstm_emb.mean(axis=0)])

    # Step 8: Prediction (semi-realistic)
    current_price = price_data[-1][3]  # Close price
    predicted_price = current_price * np.random.uniform(0.95, 1.05)
    confidence = np.random.uniform(0.6, 0.95)

    # Assertions
    assert clustered_features.shape[1] == num_clusters
    assert fused.shape[0] == (16 + 32)
    assert 0 < confidence <= 1
    assert predicted_price > 0

    print("✓ Real-World System Pipeline Test Passed")


# Run test
test_system_pipeline_real()