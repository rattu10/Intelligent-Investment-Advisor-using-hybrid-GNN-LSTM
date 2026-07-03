# Intelligent Investment Advisor using Machine Learning

## 🎯 Project Overview

This web application implements a **hybrid GNN-LSTM forecasting framework** for stock market prediction, incorporating three unique enhancements:

1. **Technical Indicators Clustering**: Groups correlated and non-correlated indicators for enhanced processing
2. **Ontology-Driven Sentiment Analysis**: Extracts structured relationships from financial news
3. **GNN-LSTM Hybrid Architecture**: Combines temporal and relational modeling for accurate predictions

## ✨ Features

- 🤖 **AI-Powered Predictions**: Deep learning models for stock price forecasting
- 📊 **Interactive Dashboard**: Real-time visualization of predictions and analysis
- 🔍 **Technical Analysis**: Clustered technical indicators with correlation analysis
- 📰 **Sentiment Analysis**: Ontology-driven financial news processing
- 🎨 **Modern UI**: Beautiful, responsive web interface
- ⚡ **Real-time Data**: Integration with financial data sources

## 🛠️ Technology Stack

- **Backend & Database**: Flask (Python), SQLite (SQLAlchemy)
- **Machine Learning**: PyTorch, scikit-learn
- **Graph Processing**: PyTorch Geometric, NetworkX
- **Sentiment Analysis**: Transformers, FinBERT
- **Data Processing**: pandas, numpy, yfinance
- **Visualization**: Plotly.js
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for fetching stock data)

## 🚀 Quick Start

### Option 1: Using Startup Scripts (Recommended)

#### On Windows:
```bash
double-click start.bat
```

#### On Linux/macOS:
```bash
chmod +x start.sh
./start.sh
```

### Option 2: Manual Installation

1. **Clone or download the project**
   ```bash
   git clone https://github.com/rattu10/Intelligent-Investment-Advisor-using-hybrid-GNN-LSTM.git
   cd stock-forecasting-gnn-lstm
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On Linux/macOS:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   On Windows (PowerShell):
   ```bash
   cd src
   $env:FLASK_APP="app_user_friendly.py"
   $env:PYTHONIOENCODING="utf-8"
   python -m flask run --host=127.0.0.1 --port=5000 --no-reload
   ```
   
   On macOS/Linux (Bash):
   ```bash
   cd src
   export FLASK_APP="app_user_friendly.py"
   export PYTHONIOENCODING="utf-8"
   python -m flask run --host=127.0.0.1 --port=5000 --no-reload
   ```

6. **Open your browser and go to:**
   ```
   http://127.0.0.1:5000
   ```

## 💻 Usage Instructions

### 1. **Stock Selection**
   - Enter a valid stock symbol (e.g., AAPL, GOOGL, MSFT)
   - Select the prediction period (5, 10, 15, or 30 days)
   - Click "Generate Prediction"

### 2. **View Results**
   - **Price Predictions**: Interactive chart showing forecasted prices
   - **Summary Cards**: Current price, predicted price, expected change, and confidence
   - **Technical Analysis**: Clustered indicators and correlation analysis
   - **Sentiment Analysis**: Overall sentiment and recent market events

### 3. **Interpret Results**
   - **Green indicators**: Positive sentiment/growth
   - **Red indicators**: Negative sentiment/decline
   - **Cluster analysis**: Shows how technical indicators are grouped
   - **Confidence score**: Model's certainty in predictions

## 🏗️ Project Structure

```
stock-forecasting/
├── src/
│   ├── app_user_friendly.py   # Main Full-Featured Flask application (Auth, Dashboard, Portfolio)
│   ├── app.py                 # Legacy/Basic single-page Flask application
│   ├── config.json            # Configuration file
│   ├── models/
│   │   ├── __init__.py
│   │   └── gnn_lstm_model.py  # GNN-LSTM implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_processor.py  # Data fetching and preprocessing
│   │   ├── technical_indicators.py # Technical analysis and clustering
│   │   ├── ontology_sentiment.py   # FinBERT Ontology sentiment analysis
│   │   └── sentiment_analyzer.py   # Basic sentiment analysis helper
│   └── templates/
│       ├── home.html          # Landing page (Welcome, stats overview)
│       ├── login.html         # User login portal (Modern light theme)
│       ├── register.html      # User registration portal
│       ├── profile_setup.html # Onboarding/Risk preference configuration
│       ├── dashboard.html     # Main dashboard analytics (Model Accuracy MAPE)
│       ├── predict.html       # Interactive GNN-LSTM stock prediction interface
│       ├── recommendations.html # Personalized stock recommendations
│       └── portfolio.html     # User portfolio tracking & USD/INR conversion
├── requirements.txt           # Python dependencies
├── requirements_simple.txt    # Lightweight dependencies (No-ML fallback)
├── requirements_stable.txt    # Pinned/Stable dependencies
├── start.bat                  # One-click Windows startup script
├── start.sh                   # One-click Unix startup script
├── benchmark_models.py        # Model benchmarking script
├── ADVANCED_RESEARCH_IMPLEMENTATION.md # Technical research mapping
└── LICENSE                    # MIT terms of use
```

## 🧠 Model Architecture

### GNN-LSTM Hybrid Framework

1. **Data Input Layer**
   - Stock price data (OHLCV)
   - Technical indicators (RSI, MACD, SMA, etc.)
   - Sentiment features from financial news

2. **Technical Indicator Clustering**
   - K-means clustering of indicators
   - Correlation analysis
   - Feature group identification

3. **Graph Neural Network (GNN)**
   - Models inter-feature relationships
   - Captures complex dependencies
   - Graph convolution operations

4. **LSTM Network**
   - Temporal pattern recognition
   - Sequential data processing
   - Long-term dependency modeling

5. **Attention Mechanism**
   - Feature importance weighting
   - Multi-head attention
   - Enhanced prediction accuracy

6. **Prediction Output**
   - Multi-day price forecasts
   - Confidence intervals
   - Trend analysis

### Sentiment Analysis Pipeline

1. **News Data Collection**
   - Financial news aggregation
   - Company-specific filtering
   - Real-time updates

2. **Ontology-Driven Processing**
   - Entity relationship extraction
   - Event classification
   - Market impact assessment

3. **Sentiment Scoring**
   - FinBERT model integration
   - Polarity and intensity analysis
   - Temporal sentiment tracking

## ⚙️ Configuration

### Model Parameters
- **GNN Layers**: 3 (configurable in `src/models/gnn_lstm_model.py`)
- **LSTM Hidden Size**: 128
- **Attention Heads**: 8
- **Sequence Length**: 30 days
- **Clusters**: 3 (technical indicators)

### Data Sources
- **Stock Data**: Yahoo Finance (yfinance)
- **News Data**: Mock data (replaceable with NewsAPI, Alpha Vantage)
- **Sentiment Model**: FinBERT (financial domain)

## 🔧 Customization

### Adding New Technical Indicators
1. Modify `src/utils/technical_indicators.py`
2. Add indicator calculation in `calculate_indicators()`
3. Update clustering logic if needed

### Integrating Real News APIs
1. Update `src/utils/sentiment_analyzer.py` or `src/utils/ontology_sentiment.py`
2. Replace `get_financial_news()` method
3. Add API keys to configuration

### Model Architecture Changes
1. Modify `src/models/gnn_lstm_model.py`
2. Adjust layer sizes, depths, or architectures
3. Update training parameters

## 📊 Example Predictions

The application provides:
- **Price forecasts** for 5-30 days ahead
- **Confidence scores** (typically 75-95%)
- **Technical analysis** with clustered indicators
- **Sentiment insights** from recent news

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   - Change port in `app_user_friendly.py`: `app.run(port=5001)`
   - Or kill existing process

3. **Data Fetching Issues**
   - Check internet connection
   - Verify stock symbol validity
   - Application falls back to mock data

4. **Model Loading Errors**
   - Ensure all dependencies are installed
   - Check available memory
   - Reduce model complexity if needed

### Performance Tips

- **First run** may be slower due to model downloads
- **Cache** is implemented for repeated requests
- **Reduce prediction days** for faster processing
- **Use mock data** mode for testing

## 🚀 Deployment

### Local Development
```bash
export FLASK_ENV=development
export FLASK_APP=app_user_friendly.py
python -m flask run --no-reload
```

### Production Deployment
```bash
gunicorn --bind 0.0.0.0:5000 app_user_friendly:app
```

### Docker Deployment
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app_user_friendly.py"]
```

## 📝 Research Paper Abstract

*"Stock market forecasting has long been a critical yet challenging problem due to the volatility, non-linear patterns, and multi-factor dependencies of financial data. Traditional deep learning approaches, such as Long Short-Term Memory (LSTM) models, have shown success in capturing temporal dependencies, while Graph Neural Networks (GNNs) have been applied to represent inter-stock relationships. However, most existing studies focus solely on correlations derived from price movements or employ sentiment analysis at a basic polarity level, leaving gaps in the integration of structured knowledge and heterogeneous feature relationships."*

This application implements the proposed hybrid framework that bridges time-series learning, graph-based reasoning, and ontology-driven sentiment integration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FinBERT**: Financial sentiment analysis model
- **PyTorch Geometric**: Graph neural network framework
- **Yahoo Finance**: Stock data source
- **Bootstrap**: UI framework
- **Plotly**: Interactive visualizations

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue with detailed description

---

**Built with ❤️ for financial research and AI enthusiasts**
