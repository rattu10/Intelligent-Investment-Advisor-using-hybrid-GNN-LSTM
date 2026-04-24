# Development and Testing Notes

## Quick Setup for VS Code

1. **Open in VS Code**: 
   - Open the project folder in VS Code
   - VS Code will automatically detect the Python environment

2. **Install Python Extension**: 
   - Install the Python extension by Microsoft if not already installed

3. **Select Python Interpreter**:
   - Press `Ctrl+Shift+P` (Cmd+Shift+P on Mac)
   - Type "Python: Select Interpreter"
   - Choose the Python interpreter in your virtual environment

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**:
   - Open terminal in VS Code (`Ctrl+` `)
   - Run: `python app.py`
   - Or use the startup scripts: `./start.sh` or `start.bat`

## Testing the Application

### Quick Test
```bash
python test_app.py --quick
```

### Full Test Suite
```bash
python test_app.py --full
```

## Development Workflow

1. **Make Changes**: Edit code in VS Code
2. **Test Changes**: Run `python test_app.py --quick`
3. **Start App**: Run `python app.py`
4. **View in Browser**: Go to `http://localhost:5000`

## Debugging in VS Code

1. **Set Breakpoints**: Click on line numbers to set breakpoints
2. **Debug Configuration**: VS Code will auto-generate debug config
3. **Start Debugging**: Press F5 to start debugging

## Common VS Code Extensions for This Project

- **Python** (Microsoft): Essential for Python development
- **Pylance** (Microsoft): Advanced Python language support
- **Python Docstring Generator**: Auto-generate docstrings
- **GitLens**: Enhanced Git capabilities
- **Live Server**: For frontend development
- **Thunder Client**: API testing (alternative to Postman)

## Project Structure in VS Code

```
📁 stock-forecasting/
├── 📄 app.py                 # Main Flask app (Run this)
├── 📄 requirements.txt       # Dependencies
├── 📄 test_app.py           # Test suite
├── 📄 config.json           # Configuration
├── 📁 models/               # ML models
├── 📁 utils/                # Utility modules  
└── 📁 templates/            # HTML templates
```

## Running in VS Code Terminal

1. **Open Terminal**: `Ctrl+` ` (backtick)
2. **Multiple Terminals**: Click `+` to create new terminals
3. **Split Terminal**: Click split icon

### Terminal Commands:
```bash
# Install dependencies
pip install -r requirements.txt

# Run quick test
python test_app.py --quick

# Start application
python app.py

# Run specific test
python -m unittest test_app.TestDataProcessor
```

## Hot Reload Development

The Flask app runs in debug mode by default, so:
- **Automatic Reload**: Changes to Python files trigger automatic restart
- **Error Display**: Detailed error pages in browser
- **Interactive Debugger**: Click on traceback for interactive debugging

## Browser Development

1. **Open Browser**: Go to `http://localhost:5000`
2. **Developer Tools**: Press F12
3. **Network Tab**: Monitor API calls
4. **Console Tab**: View JavaScript errors/logs

## API Testing

Test API endpoints directly:

```bash
# Test prediction endpoint
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 5}'

# Test stock search
curl http://localhost:5000/api/stocks/search?q=AAPL
```

## Performance Monitoring

- **Flask Debug Toolbar**: Add to monitor request performance
- **Python Profiler**: Use cProfile for performance analysis
- **Memory Usage**: Monitor with memory_profiler

## Troubleshooting in VS Code

### Common Issues:

1. **Import Errors**:
   - Check Python interpreter selection
   - Verify virtual environment activation
   - Install missing packages

2. **Port Issues**:
   - Change port in app.py if 5000 is occupied
   - Check if other Flask apps are running

3. **Module Not Found**:
   - Ensure project root is in Python path
   - Check file naming and structure

### Debug Steps:

1. **Check Terminal Output**: Look for error messages
2. **Use Print Statements**: Add debug prints
3. **Use VS Code Debugger**: Set breakpoints and step through
4. **Check Browser Console**: For frontend issues
5. **Review Logs**: Check Flask app logs

## Code Organization Tips

- **Models**: Keep ML models in `models/` directory
- **Utils**: Utility functions in `utils/` directory
- **Templates**: HTML files in `templates/` directory
- **Static Files**: CSS/JS in `static/` directory (if added)
- **Tests**: Test files with `test_` prefix

## Git Integration in VS Code

1. **Source Control**: Click source control icon (Ctrl+Shift+G)
2. **Stage Changes**: Click `+` next to files
3. **Commit**: Add message and commit
4. **Push/Pull**: Use command palette or status bar

This setup provides a complete development environment for the stock forecasting application in VS Code!
