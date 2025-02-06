# Smart Portfolio Manager 🚀

A full-stack application for portfolio analysis and automatic rebalancing using modern portfolio theory and artificial intelligence.

## Features ✨

### Portfolio Analysis
- Portfolio optimization based on Modern Portfolio Theory
- Risk and risk-adjusted return analysis (Sharpe, Sortino, Treynor)
- Advanced metrics calculation (VaR, CVaR, Maximum Drawdown)
- Technical analysis and market trends
- Market regime and volatility detection
- Correlation and diversification analysis
- Stress testing and risk scenarios

### Artificial Intelligence
- AI-powered portfolio analysis
- Personalized recommendations in English and Spanish
- Market regime detection
- Sentiment and trend analysis
- Portfolio optimization suggestions
- Short, medium, and long-term market predictions

### Investment Management
- Automatic rebalancing through Alpaca Trading API
- Historical performance tracking
- Customizable risk alerts
- Sector and geographic exposure analysis
- Risk concentration monitoring

### Data Visualization
- Interactive asset allocation charts
- Market trend visualization
- Real-time technical indicators
- Customizable dashboard
- Risk and return visual analysis
- Modern animations and visual effects

## Tech Stack 🛠️

### Backend
- FastAPI
- SQLAlchemy
- PyPortfolioOpt
- Pandas & NumPy
- Scikit-learn
- Alpaca Trading API
- YFinance
- Python-dotenv

### Frontend
- React + TypeScript
- Tailwind CSS
- Framer Motion
- Chart.js
- TsParticles
- Radix UI
- Axios

## Prerequisites 📋

- Python 3.8+
- Node.js 14+
- Alpaca Trading account with API keys
- Polygon.io account (optional, for additional market data)

## Installation 🔧

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/SmartPortfolio.git
cd SmartPortfolio
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your API keys
```

### 3. Frontend Setup

```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev
```

### 4. Initialize Database

```bash
cd backend
python app/init_db.py
```

## Usage 💡

### Start Backend
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend
```bash
cd frontend
npm run dev
```

### Access the Application
- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

## API Endpoints 🔌

### Portfolio Analysis
- `POST /analyze-portfolio`: Analyzes and optimizes a portfolio
- `POST /rebalance-portfolio`: Executes portfolio rebalancing
- `GET /market-analysis`: Gets current market analysis
- `POST /get-ticker-suggestions`: Gets AI-based ticker suggestions

### User Management
- `POST /register`: Registers a new user
- `POST /login`: Logs in a user
- `GET /portfolio-history`: Gets portfolio history

## Environment Variables 🔐

```env
# API Keys
POLYGON_API_KEY=your_polygon_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Database
DATABASE_URL=sqlite:///./portfolio.db

# Security
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Proxy Configuration (optional)
SOCKS_PROXY_HOST=localhost
SOCKS_PROXY_PORT=9050
USE_TOR_PROXY=true
```

## Features in Development 🚧

- [ ] Strategy backtesting
- [ ] Integration with more brokers
- [ ] Advanced market sentiment analysis
- [ ] Email/push notifications
- [ ] Multi-objective optimization
- [ ] Machine Learning for market prediction
- [ ] Cryptocurrency support
- [ ] ESG factor analysis
- [ ] Integration with additional data sources

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Contact 📧

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/SmartPortfolio](https://github.com/yourusername/SmartPortfolio)

## Acknowledgments 🙏

- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)
- [Alpaca Markets](https://alpaca.markets/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Framer Motion](https://www.framer.com/motion/)
- [Radix UI](https://www.radix-ui.com/) 