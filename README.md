<p align="center">
  <img src="https://via.placeholder.com/200x200?text=Smart+Portfolio" alt="Smart Portfolio Logo" width="200" height="200">
</p>

<h1 align="center">Smart Portfolio Manager 📊💰</h1>

<p align="center">
  <a href="https://organica-ai-solutions.github.io/SmartPorfolioWeb/"><img src="https://img.shields.io/badge/demo-live%20preview-brightgreen.svg" alt="Live Demo"></a>
  <a href="https://github.com/Organica-Ai-Solutions/SmartPorfolioWeb/blob/master/LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/version-1.0.0-orange.svg" alt="Version">
  <img src="https://img.shields.io/badge/built%20with-AI-8A2BE2" alt="Built with AI">
</p>

<p align="center">
  <b>A full-stack application for portfolio analysis and automatic rebalancing using modern portfolio theory and artificial intelligence</b>
</p>

<p align="center">
  <a href="#demo">Demo</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#api-endpoints">API Endpoints</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a> •
  <a href="#acknowledgments">Acknowledgments</a>
</p>

## Demo 🌐

### Live Demo

- **Frontend:** [https://organica-ai-solutions.github.io/SmartPorfolioWeb/](https://organica-ai-solutions.github.io/SmartPorfolioWeb/)
- **Backend API:** [https://smartportfolio-backend.onrender.com](https://smartportfolio-backend.onrender.com)
- **API Documentation:** [https://smartportfolio-backend.onrender.com/docs](https://smartportfolio-backend.onrender.com/docs)

### Screenshot

<p align="center">
  <img src="https://via.placeholder.com/800x450?text=Smart+Portfolio+Screenshot" alt="Smart Portfolio Screenshot">
</p>

## Key Features ✨

### Portfolio Analysis
- **Modern Portfolio Theory** - Efficient frontier optimization, optimal asset allocation
- **Advanced Risk Metrics** - Sharpe, Sortino, Treynor ratios, VaR, CVaR, Maximum Drawdown
- **Market Analysis** - Regime detection, volatility analysis, correlation analysis
- **Stress Testing** - Multiple risk scenarios, historical event simulations
- **Performance Tracking** - Historical performance visualization and analysis

### AI-Powered Insights
- **DeepSeek AI Integration** - For sophisticated portfolio analysis and recommendations
- **Bilingual Support** - Personalized recommendations in English and Spanish
- **Secure Data Processing** - Anonymization of sensitive financial data
- **Predictive Analysis** - Market regime prediction, trend detection, sentiment analysis
- **Custom Recommendations** - Tailored to user's risk profile and investment goals

### Smart Rebalancing
- **Alpaca Trading API Integration** - Automated portfolio rebalancing
- **Intelligent Execution** - Optimal order execution with minimal market impact
- **Tax-Efficiency** - Considers tax implications in rebalancing decisions
- **Rebalance Explanations** - AI-generated explanations for every rebalancing action
- **Commission Optimization** - Minimizes trading costs during rebalancing

### Security Features
- **Data Encryption** - Sensitive information encrypted at rest and in transit
- **Optional Tor Proxy** - Enhanced privacy for AI API calls
- **Secure Authentication** - JWT-based authentication with proper encryption
- **Environment Isolation** - Separate development and production environments
- **Secure API Design** - Rate limiting, input validation, and proper error handling

## Tech Stack 🛠️

### Backend
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white" alt="SQLAlchemy">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</p>

### Frontend
<p>
  <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS">
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chart.js&logoColor=white" alt="Chart.js">
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite">
  <img src="https://img.shields.io/badge/Framer_Motion-0055FF?style=for-the-badge&logo=framer&logoColor=white" alt="Framer Motion">
</p>

### Deployment & CI/CD
<p>
  <img src="https://img.shields.io/badge/GitHub_Pages-222222?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Pages">
  <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render">
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions">
</p>

## Architecture 🏗️

The Smart Portfolio application follows a modern, decoupled architecture:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│   Frontend      │◄────►│   Backend API   │◄────►│  External APIs  │
│   (React)       │      │   (FastAPI)     │      │  & Services     │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
       ▲                        ▲
       │                        │
       │                        │
       │                        ▼
┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │
│   GitHub Pages  │      │   Database      │
│   Hosting       │      │   (PostgreSQL)  │
│                 │      │                 │
└─────────────────┘      └─────────────────┘
```

- **Frontend**: React application with TypeScript and Tailwind CSS
- **Backend**: FastAPI with SQLAlchemy ORM
- **Database**: PostgreSQL database for user data and portfolio history
- **External Services**: DeepSeek AI API for advanced analysis, Alpaca for trading
- **Deployment**: GitHub Pages (frontend) and Render (backend)

## Installation 🔧

### Prerequisites
- Python 3.8+
- Node.js 14+
- Alpaca Trading account with API keys
- DeepSeek AI API key
- Tor (optional, for enhanced privacy)

### 1. Clone the Repository
```bash
git clone https://github.com/Organica-Ai-Solutions/SmartPorfolioWeb.git
cd SmartPorfolioWeb
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example and edit)
cp .env.example .env
```

Configure your `.env` file with necessary API keys and settings.

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment files
cp .env.example .env.development
```

Edit the `.env.development` file to set the API URL for local development.

### 4. Start Development Servers

Backend:
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Frontend:
```bash
cd frontend
npm run dev
```

## Deployment 🚀

The application is deployed using a combination of services:

### Backend Deployment (Render)

1. Fork or clone the repository
2. Create a Render account and connect to your GitHub repository
3. Create a new Web Service with the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Environment Variables: Set up all required variables (API keys, etc.)

### Frontend Deployment (GitHub Pages)

1. Update the API URL in `.env.production` to point to your backend
2. Build and deploy with:
   ```bash
   cd frontend
   npm run build
   npm run deploy
   ```

## API Endpoints 🔌

### Portfolio Analysis
- `POST /analyze-portfolio-simple` - Analyzes a portfolio with basic metrics
- `POST /ai-portfolio-analysis` - AI-powered portfolio analysis
- `POST /ai-rebalance-explanation` - Gets AI explanation for rebalancing
- `POST /rebalance-portfolio-simple` - Rebalances portfolio with Alpaca

### Market Data
- `GET /market-regimes` - Current market regime detection
- `GET /market-analysis` - Comprehensive market analysis
- `GET /ticker-data/{ticker}` - Historical data for specific ticker

### Users & Authentication
- `POST /register` - Register new user
- `POST /login` - User login
- `GET /users/me` - Get current user data
- `PUT /users/me` - Update user data
- `GET /users/portfolios` - Get user portfolios

### Full API Documentation
For the complete API documentation, visit the [Swagger UI](https://smartportfolio-backend.onrender.com/docs) or [ReDoc](https://smartportfolio-backend.onrender.com/redoc).

## Environment Variables 🔐

### Backend

```env
# API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1
POLYGON_API_KEY=your_polygon_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Database
DATABASE_URL=postgresql://user:password@host:port/database

# Security
ENCRYPTION_KEY=your_encryption_key
SECRET_KEY=your_jwt_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Proxy Configuration
SOCKS_PROXY_HOST=localhost
SOCKS_PROXY_PORT=9050
USE_TOR_PROXY=true
```

### Frontend

```env
# Backend API URL
VITE_API_URL=https://your-backend-url.com
```

## Roadmap 🚧

### Short-term Goals
- [ ] Mobile-responsive design improvements
- [ ] Additional portfolio optimization strategies
- [ ] Enhanced AI explanations for investment decisions
- [ ] User feedback system

### Medium-term Goals
- [ ] Advanced backtesting engine
- [ ] Integration with more brokers (IBKR, TDAmeritrade)
- [ ] Real-time market alerts
- [ ] Social sharing features for portfolios

### Long-term Goals
- [ ] Machine learning models for enhanced predictions
- [ ] Cryptocurrency portfolio management
- [ ] Advanced factor analysis (ESG, momentum, quality)
- [ ] Custom investment strategy builder
- [ ] Mobile applications (iOS, Android)

## Contributing 🤝

We welcome contributions to the Smart Portfolio project!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the code style guidelines.

## License 📄

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments 🙏

- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) - Portfolio optimization library
- [Alpaca Markets](https://alpaca.markets/) - Commission-free trading API
- [DeepSeek AI](https://www.deepseek.com/) - Advanced AI capabilities
- [FastAPI](https://fastapi.tiangolo.com/) - Modern API framework
- [React](https://reactjs.org/) - Frontend framework
- [Tailwind CSS](https://tailwindcss.com/) - CSS framework
- [Framer Motion](https://www.framer.com/motion/) - Animation library
- [Chart.js](https://www.chartjs.org/) - Data visualization
- [Radix UI](https://www.radix-ui.com/) - Headless UI components 