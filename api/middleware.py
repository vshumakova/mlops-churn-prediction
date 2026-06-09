from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

def add_security_middleware(app):
    app.add_middleware(HTTPSRedirectMiddleware)
    
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*.github.dev", "localhost", "127.0.0.1"]
    )
