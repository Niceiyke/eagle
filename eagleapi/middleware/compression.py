# middleware/compression.py
"""Compression middleware for Eagle framework."""
from .base import EagleMiddleware
from fastapi.responses import Response
from fastapi.requests import Request
import gzip
import io

class CompressionMiddleware(EagleMiddleware):
    """Gzip compression middleware."""
    
    def setup(self):
        self.minimum_size = self.config.get('minimum_size', 1024)
        self.compressible_types = self.config.get('compressible_types', [
            'application/json',
            'text/html',
            'text/css',
            'text/javascript',
            'application/javascript'
        ])
    
    async def after_response(self, request: Request, response: Response) -> Response:
        """Compress response if applicable."""
        # Check if client supports gzip
        accept_encoding = request.headers.get('accept-encoding', '')
        if 'gzip' not in accept_encoding:
            return response
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not any(ct in content_type for ct in self.compressible_types):
            return response
        
        # Check content length
        if hasattr(response, 'body') and len(response.body) < self.minimum_size:
            return response
        
        # Compress the response
        if hasattr(response, 'body'):
            compressed = gzip.compress(response.body)
            if len(compressed) < len(response.body):
                response.body = compressed
                response.headers['content-encoding'] = 'gzip'
                response.headers['content-length'] = str(len(compressed))
        
        return response