# SmartPortfolio 📈

A modern, AI-powered portfolio management system that helps investors make data-driven decisions.

## Features 🚀

- **Smart Portfolio Analysis** - Advanced analytics and performance metrics
- **AI-Driven Insights** - Powered by DeepSeek AI for market analysis
- **Real-Time Data** - Integration with Polygon.io and Alpaca Markets
- **Interactive Charts** - Beautiful visualizations of portfolio performance
- **Secure Authentication** - JWT-based authentication system
- **Responsive Design** - Works seamlessly on desktop and mobile

## Documentation 📚

- [Portfolio Optimization Guide](docs/portfolio_optimization.md) - Detailed guide on portfolio optimization techniques
- [API Documentation](backend/README.md) - Backend API documentation
- [Frontend Guide](frontend/README.md) - Frontend development guide
- [Security Guidelines](SECURITY.md) - Security best practices

## Tech Stack 💻

### Frontend
- React with TypeScript
- Vite for build tooling
- TailwindCSS for styling
- Chart.js for visualizations

### Backend
- FastAPI (Python)
- SQLAlchemy ORM
- PostgreSQL/SQLite
- JWT Authentication

### APIs & Services
- Alpaca Markets API for trading
- Polygon.io for market data
- DeepSeek AI for market analysis
- CoinMarketCap for crypto data

## Getting Started 🏁

### Prerequisites
- Node.js 18+
- Python 3.9+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Organica-Ai-Solutions/SmartPorfolioWeb.git
cd SmartPorfolioWeb
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Create your env file
```

3. Set up the frontend:
```bash
cd frontend
npm install
cp .env.example .env  # Create your env file
```

4. Configure your environment variables:
- Copy `.env.example` to `.env` in both frontend and backend directories
- Fill in your API keys and configuration values

### Running the Application

1. Start the backend:
```bash
cd backend
uvicorn app.main:app --reload
```

2. Start the frontend:
```bash
cd frontend
npm run dev
```

Visit `http://localhost:5173` to see the application.

## Contributing 🤝

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Security 🔒

- Never commit API keys or sensitive data
- Use environment variables for secrets
- See [Security Guidelines](SECURITY.md) for best practices

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 👏

- [Alpaca Markets](https://alpaca.markets/) for trading infrastructure
- [Polygon.io](https://polygon.io/) for market data
- [DeepSeek](https://deepseek.com/) for AI capabilities

---

Made with ❤️ by [Organica AI Solutions](https://github.com/Organica-Ai-Solutions) 