from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import os

def add_security_middleware(app):
    if os.environ.get('ENABLE_HTTPS', 'false').lower() == 'true':
        app.add_middleware(HTTPSRedirectMiddleware)
    
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=[
            "*.github.dev",
            "*.app.github.dev", 
            "localhost", 
            "127.0.0.1"
        ]
    )
