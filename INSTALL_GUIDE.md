# 🚀 INSTALLATION & USAGE GUIDE

## Quick Start for VS Code

### 1. Setup in VS Code

1. **Open Project in VS Code**
   - Open the project folder in VS Code
   - Open terminal in VS Code: `Ctrl+` ` (backtick)

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   ```

3. **Activate Virtual Environment**
   - **Windows**: `venv\Scripts\activate`
   - **Mac/Linux**: `source venv/bin/activate`

4. **Install Dependencies**
   ```bash
   # Essential dependencies (minimum working setup)
   pip install flask flask-cors pandas numpy yfinance ta scikit-learn
   pip install jinja2 werkzeug blinker itsdangerous markupsafe
   
   # Optional: For full ML features (large download)
   pip install torch transformers torch-geometric sentence-transformers
   
   # Optional: For enhanced visualizations
   pip install plotly matplotlib seaborn networkx
   ```

5. **Run the Application**
   ```bash
   python app_simple.py
   ```

6. **Open in Browser**
   - Go to: `http://localhost:5000`

---

## Alternative Installation Methods

### Method 1: Using Requirements File
```bash
pip install -r requirements.txt
```
*Note: This installs all dependencies including PyTorch (large download)*

### Method 2: Minimal Setup (Fastest)
```bash
pip install flask flask-cors pandas numpy yfinance ta
python app_simple.py
```

### Method 3: Using Startup Scripts
- **Windows**: Double-click `start.bat`
- **Mac/Linux**: `./start.sh`

---

## Testing the Installation

### Quick Test
```bash
python basic_test.py
```

### Full Demo
```bash
python demo.py
```

### Check Application Status
```bash
# After starting the app, check:
curl http://localhost:5000/api/status
```

---

## Available Applications

### 1. **app_simple.py** (Recommended for beginners)
- ✅ Works with minimal dependencies
- ✅ Automatic fallbacks for missing components
- ✅ Real stock data integration
- ⚠️ Simplified ML models (no PyTorch required)

```bash
python app_simple.py
```

### 2. **app.py** (Full featured)
- ✅ Complete GNN-LSTM implementation
- ✅ Advanced sentiment analysis
- ✅ Full PyTorch integration
- ❗ Requires all dependencies

```bash
python app.py
```

---

## Usage Instructions

### Web Interface

1. **Start Application**
   ```bash
   python app_simple.py
   ```

2. **Open Browser**: `http://localhost:5000`

3. **Use the Interface**:
   - Enter stock symbol (e.g., AAPL, GOOGL, MSFT)
   - Select prediction period (5-30 days)
   - Click "Generate Prediction"
   - View results, charts, and analysis

### API Endpoints

```bash
# Get stock prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 5}'

# Search stocks
curl http://localhost:5000/api/stocks/search?q=AAPL

# Get technical analysis
curl http://localhost:5000/api/technical-analysis/AAPL

# Get sentiment analysis
curl http://localhost:5000/api/sentiment/AAPL

# Check app status
curl http://localhost:5000/api/status
```

---

## Features Available

### ✅ Always Available
- **Stock Data Fetching**: Real-time data from Yahoo Finance
- **Technical Indicators**: 15+ indicators with clustering
- **Price Predictions**: Multi-day forecasts
- **Interactive Dashboard**: Modern web interface
- **Responsive Design**: Works on desktop and mobile

### 🔄 Conditional Features
- **GNN-LSTM Model**: Requires PyTorch
- **Advanced Sentiment**: Requires Transformers
- **Enhanced Visualizations**: Requires additional libraries

---

## Supported Stock Symbols

The application works with any valid stock symbol, including:

**Popular US Stocks**:
- AAPL (Apple), GOOGL (Google), MSFT (Microsoft)
- AMZN (Amazon), TSLA (Tesla), META (Meta)
- NVDA (NVIDIA), JPM (JPMorgan), V (Visa)

**International Stocks**:
- Add country suffix: `1211.HK` (Hong Kong), `600519.SS` (Shanghai)

---

## Troubleshooting

### Common Issues & Solutions

1. **Port 5000 in use**
   ```bash
   # Change port in app_simple.py:
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

2. **Module not found errors**
   ```bash
   pip install [missing-module]
   # Or use app_simple.py which has fallbacks
   ```

3. **Data fetching errors**
   - Check internet connection
   - Verify stock symbol is valid
   - App will use mock data if needed

4. **Memory issues with PyTorch**
   ```bash
   # Use simplified version:
   python app_simple.py
   ```

5. **VS Code not detecting Python**
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose your virtual environment

### Performance Tips

- **First run**: May be slower due to data downloads
- **Stock symbols**: Use uppercase (AAPL not aapl)
- **Prediction period**: Shorter periods = faster processing
- **Internet**: Required for real stock data

---

## Development in VS Code

### Recommended Extensions
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **Python Docstring Generator**
- **GitLens**
- **Thunder Client** (for API testing)

### Debugging
1. Set breakpoints by clicking line numbers
2. Press `F5` to start debugging
3. Use terminal for live testing: `Ctrl+` `

### File Structure
```
📁 stock-forecasting/
├── 📄 app_simple.py         # Simplified app (recommended)
├── 📄 app.py               # Full-featured app
├── 📄 requirements.txt     # All dependencies
├── 📄 basic_test.py        # Quick functionality test
├── 📄 demo.py              # Full demo script
├── 📁 models/              # ML models
├── 📁 utils/               # Utility modules
└── 📁 templates/           # Web interface
```

---

## Next Steps

1. **Start with basic setup**: Use `app_simple.py`
2. **Test functionality**: Run `python basic_test.py`
3. **Explore the demo**: Run `python demo.py`
4. **Use the web interface**: Open browser to `http://localhost:5000`
5. **Experiment with different stocks**: Try various symbols
6. **Customize**: Modify code for your specific needs

---

## Support & Documentation

- **README.md**: Complete project overview
- **DEVELOPMENT.md**: Advanced development guide
- **config.json**: Configuration settings
- **test_app.py**: Comprehensive test suite

---

**🎉 You're ready to start forecasting stock prices with AI!**