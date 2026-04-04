# DEPLOYMENT INSTRUCTIONS FOR CLAUDE CODE

## What this is
A Flask web app (control panel + pipeline runner) for an AI intelligence pipeline.
Deploy to your VPS (e.g. Hetzner) at root@YOUR_SERVER_IP.

## Server already has
- Ubuntu 24.04
- nginx (running)
- Python 3.12 venv at /opt/ai-intel/venv
- gunicorn installed in venv
- systemd service: ai-intel
- nginx config: /etc/nginx/sites-available/ai-intel

## What to deploy
1. Copy app.py and pipeline_runner.py to /opt/ai-intel/ on the server (replace existing app.py)
2. Copy .env.example to /opt/ai-intel/.env.example
3. Create /opt/ai-intel/.env with actual values (ask me for the passwords/keys)
4. Ensure /opt/ai-intel/data/daily and /opt/ai-intel/data/weekly directories exist
5. Restart the service: systemctl restart ai-intel
6. Verify: curl http://localhost:5000/ should redirect to /login

## Gunicorn config
The existing systemd service at /etc/systemd/system/ai-intel.service should run:
/opt/ai-intel/venv/bin/gunicorn --workers 2 --bind 127.0.0.1:5000 app:app
Working directory: /opt/ai-intel

## After deployment
Visit http://YOUR_SERVER_IP — should show login page.
Login with ADMIN_USERNAME and ADMIN_PASSWORD from .env.
