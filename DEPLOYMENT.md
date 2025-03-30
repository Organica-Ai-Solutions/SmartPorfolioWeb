# Deployment Guide

This guide provides instructions for deploying the SmartPortfolio application in a production environment.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A production server (e.g., Linux-based VPS)
- Domain name (optional but recommended)
- SSL certificate (required for production)

## Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SmartPortfolio.git
   cd SmartPortfolio
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cd backend
   cp .env.example .env
   ```
   Edit `.env` with your production settings:
   - Set `ENV=production`
   - Configure your Alpaca API keys
   - Set proper CORS origins
   - Configure rate limiting
   - Set allowed hosts
   - Add your API keys

## Production Configuration

1. Update CORS settings in `.env`:
   ```
   CORS_ORIGINS=https://your-frontend-domain.com
   ```

2. Configure allowed hosts:
   ```
   ALLOWED_HOSTS=["your-domain.com", "api.your-domain.com"]
   ```

3. Set up rate limiting:
   ```
   RATE_LIMIT_PER_MINUTE=60
   ```

4. Configure API security:
   ```
   API_KEY_HEADER=X-API-Key
   ALLOWED_API_KEYS=your-secret-api-key-1,your-secret-api-key-2
   ```

## Running with Gunicorn

1. Install Gunicorn:
   ```bash
   pip install gunicorn
   ```

2. Create a Gunicorn configuration file `gunicorn_config.py`:
   ```python
   import multiprocessing
   import os

   # Server socket
   bind = f"{os.getenv('API_HOST', '0.0.0.0')}:{os.getenv('API_PORT', '8080')}"
   backlog = 2048

   # Worker processes
   workers = multiprocessing.cpu_count() * 2 + 1
   worker_class = 'uvicorn.workers.UvicornWorker'
   worker_connections = 1000
   timeout = 60
   keepalive = 2

   # Logging
   accesslog = '-'
   errorlog = '-'
   loglevel = 'info'
   ```

3. Start the application:
   ```bash
   gunicorn -c gunicorn_config.py app.main:app
   ```

## Nginx Configuration

1. Install Nginx:
   ```bash
   sudo apt update
   sudo apt install nginx
   ```

2. Create Nginx configuration:
   ```nginx
   server {
       listen 80;
       server_name api.your-domain.com;

       location / {
           proxy_pass http://localhost:8080;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

3. Enable SSL with Certbot:
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d api.your-domain.com
   ```

## Process Management

1. Create a systemd service file `/etc/systemd/system/smartportfolio.service`:
   ```ini
   [Unit]
   Description=SmartPortfolio API
   After=network.target

   [Service]
   User=your-user
   Group=your-group
   WorkingDirectory=/path/to/SmartPortfolio/backend
   Environment="PATH=/path/to/SmartPortfolio/backend/venv/bin"
   ExecStart=/path/to/SmartPortfolio/backend/venv/bin/gunicorn -c gunicorn_config.py app.main:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

2. Enable and start the service:
   ```bash
   sudo systemctl enable smartportfolio
   sudo systemctl start smartportfolio
   ```

## Monitoring

1. Install monitoring tools:
   ```bash
   pip install prometheus_client
   ```

2. Monitor logs:
   ```bash
   sudo journalctl -u smartportfolio -f
   ```

## Security Checklist

- [ ] SSL/TLS enabled
- [ ] API keys configured
- [ ] Rate limiting enabled
- [ ] CORS origins restricted
- [ ] Firewall configured
- [ ] Regular security updates
- [ ] Monitoring in place
- [ ] Backup strategy implemented

## Troubleshooting

1. Check application logs:
   ```bash
   sudo journalctl -u smartportfolio -f
   ```

2. Check Nginx logs:
   ```bash
   sudo tail -f /var/log/nginx/error.log
   sudo tail -f /var/log/nginx/access.log
   ```

3. Test API health:
   ```bash
   curl https://api.your-domain.com/health
   curl https://api.your-domain.com/readiness
   ```

## Backup and Recovery

1. Database backups (if applicable):
   ```bash
   # Set up regular backups
   0 0 * * * /path/to/backup-script.sh
   ```

2. Environment configuration backup:
   ```bash
   # Backup .env file
   cp .env .env.backup
   ```

## Updates and Maintenance

1. Update application:
   ```bash
   git pull origin main
   pip install -r requirements.txt
   sudo systemctl restart smartportfolio
   ```

2. Monitor system resources:
   ```bash
   htop
   df -h
   ```

## Support

For issues and support:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation 