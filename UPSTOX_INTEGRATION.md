# Upstox Integration Guide for BTSM

This guide provides detailed instructions on how to set up the Upstox API integration with the Backtesting & Trading Strategy Manager (BTSM) system.

## Prerequisites

1. Upstox Trading Account 
2. Upstox Developer Account with API access
3. Python 3.8+ with required packages installed
4. BTSM codebase cloned to your local system

## Step 1: Obtain Upstox API Credentials

1. Visit the [Upstox Developer Portal](https://developer.upstox.com/)
2. Register/Login to your Upstox developer account
3. Create a new application with the following details:
   - Name: BTSM Trading System (or your preferred name)
   - Redirect URL: `http://localhost:5000/callback`
   - Permissions: Request all required trading permissions
4. After creating the application, you'll receive:
   - API Key
   - API Secret
   - Note these credentials for later use

## Step 2: Configure Environment Variables

1. Create a `.env` file in the root directory of the BTSM project:

```
UPSTOX_API_KEY=your_upstox_api_key_here
UPSTOX_API_SECRET=your_upstox_api_secret_here
UPSTOX_REDIRECT_URL=http://localhost:5000/callback
```

2. Update the `config/config.ini` file with your Upstox credentials:

```ini
[brokers]
# Other broker configurations...

# Upstox API credentials
upstox_api_key = your_upstox_api_key_here
upstox_api_secret = your_upstox_api_secret_here
```

## Step 3: Setting Up the Redirect URL Handler

For live trading with Upstox, you need to implement a simple web server to handle the OAuth redirection. Below is a simple Flask implementation:

1. Install Flask if not already installed:
```bash
pip install flask
```

2. Create a file named `upstox_auth.py` in the project root:

```python
from flask import Flask, request
import webbrowser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get credentials from environment variables
api_key = os.getenv("UPSTOX_API_KEY")
redirect_url = os.getenv("UPSTOX_REDIRECT_URL")

@app.route("/")
def home():
    """Home page with auth link"""
    auth_url = f"https://api.upstox.com/v2/login/authorization?client_id={api_key}&redirect_uri={redirect_url}&response_type=code"
    return f'<a href="{auth_url}">Authenticate with Upstox</a>'

@app.route("/callback")
def callback():
    """Callback endpoint that receives the authorization code"""
    code = request.args.get('code')
    if code:
        # Store the code in a file for the main application to use
        with open("upstox_auth_code.txt", "w") as f:
            f.write(code)
        return f"Authentication successful! Authorization code: {code}<br>You can close this window and return to the application."
    else:
        return "Authentication failed! No authorization code received."

if __name__ == "__main__":
    # Open browser automatically
    webbrowser.open('http://localhost:5000/')
    app.run(port=5000)
```

3. Run the authentication server when you need to generate a new token:
```bash
python upstox_auth.py
```

4. Follow the browser prompts to authenticate with Upstox
5. After successful authentication, the authorization code will be saved to `upstox_auth_code.txt`

## Step 4: Integrate with the Trading System

The BTSM system already includes the `UpstoxBroker` class that handles the API communication. However, to use the authentication code:

1. Modify the `connect()` method in `/execution/upstox.py` to use the authentication code:

```python
def connect(self) -> bool:
    """Connect to Upstox API."""
    if not self.api_key or not self.api_secret:
        self.logger.error("API key or secret is missing")
        return False
    
    try:
        # Check if auth code file exists
        if os.path.exists("upstox_auth_code.txt"):
            with open("upstox_auth_code.txt", "r") as f:
                auth_code = f.read().strip()
                
            # Get access token using auth code
            token_url = "https://api.upstox.com/v2/login/authorization/token"
            data = {
                "code": auth_code,
                "client_id": self.api_key,
                "client_secret": self.api_secret,
                "redirect_uri": self.config.get('upstox_redirect_url', 'http://localhost:5000/callback'),
                "grant_type": "authorization_code"
            }
            
            response = requests.post(token_url, data=data)
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get("access_token")
                
                # Set default headers for API requests
                self.session.headers.update({
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.access_token}'
                })
                
                # Load instruments
                self._load_instruments()
                
                self.connected = True
                self.logger.info("Connected to Upstox API")
                return True
            else:
                self.logger.error(f"Failed to get access token: {response.text}")
                return False
        else:
            self.logger.error("Authentication code not found. Run upstox_auth.py first")
            return False
            
    except Exception as e:
        self.logger.error(f"Error connecting to Upstox API: {str(e)}")
        return False
```

## Step 5: Running the System with Upstox

### Backtesting Mode
```bash
python main.py --mode backtest --broker upstox --strategy moving_average --report
```

### Paper Trading Mode
```bash
python main.py --mode paper_trading --broker upstox --strategy moving_average
```

### Live Trading Mode
```bash
# First run the authentication server
python upstox_auth.py

# After authenticating in browser, run the trading system
python main.py --mode live --broker upstox --strategy moving_average
```

## Troubleshooting

1. **Authentication Issues**
   - Verify your API key and secret in both `.env` and `config.ini`
   - Ensure your redirect URL matches what's registered in the Upstox developer portal
   - Check if your authentication token is still valid (they expire after a certain period)

2. **Connection Errors**
   - Check your internet connection
   - Verify that the Upstox API is currently available
   - Ensure you've authenticated using the proper credentials

3. **Trading Issues**
   - Verify that your Upstox account has sufficient funds
   - Ensure the symbols you're trying to trade are available in your market segment
   - Check if your risk management settings are blocking trades

## Additional Resources

- [Upstox API Documentation](https://upstox.com/developer/api-documentation/)
- [BTSM Documentation](README.md)
- [Troubleshooting Guide](SETUP_INSTRUCTIONS.md#troubleshooting)

## Security Considerations

1. Never commit your API credentials to version control
2. Store sensitive information in environment variables or secure configuration files
3. Implement proper error handling to avoid exposing sensitive information in logs
4. Use HTTPS for all API communications
5. Regularly rotate your API credentials