import multiprocessing
import os

# Server socket
bind = f"{os.getenv('API_HOST', '0.0.0.0')}:{os.getenv('API_PORT', '8080')}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000
timeout = int(os.getenv('API_TIMEOUT', '60'))
keepalive = 2

# SSL (uncomment if not using Nginx for SSL termination)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'
# ssl_version = 'TLS'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = os.getenv('LOG_LEVEL', 'info').lower()

# Process naming
proc_name = 'smartportfolio'
default_proc_name = 'smartportfolio'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Logging
logconfig = None
syslog = False
syslog_addr = 'udp://localhost:514'
syslog_prefix = None
syslog_facility = 'user'
enable_stdio_inheritance = False

# Process management
preload_app = True
reload = False
reload_engine = 'auto'
spew = False
check_config = False

# Server hooks
def on_starting(server):
    """
    Called just before the master process is initialized.
    """
    pass

def on_reload(server):
    """
    Called before the worker processes are forked.
    """
    pass

def when_ready(server):
    """
    Called just after the server is started.
    """
    pass

def pre_fork(server, worker):
    """
    Called just before a worker is forked.
    """
    pass

def post_fork(server, worker):
    """
    Called just after a worker has been forked.
    """
    pass

def pre_exec(server):
    """
    Called just before a new master process is forked.
    """
    pass

def pre_request(worker, req):
    """
    Called just before a request is processed.
    """
    worker.log.debug("%s %s" % (req.method, req.path))

def post_request(worker, req, environ, resp):
    """
    Called after a request is processed.
    """
    pass

def worker_int(worker):
    """
    Called just after a worker exited on SIGINT or SIGQUIT.
    """
    pass

def worker_abort(worker):
    """
    Called when a worker received the SIGABRT signal.
    """
    pass

def worker_exit(server, worker):
    """
    Called just after a worker has been exited.
    """
    pass 