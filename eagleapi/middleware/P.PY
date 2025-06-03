




    







# Updated main __init__.py integration
class EagleAPI(FastAPI):
    """Main application class for the Eagle framework."""
    
    def __init__(self, *args, **kwargs):
        # Extract middleware config
        self.middleware_config = kwargs.pop('middleware_config', {})
        self.cache_config = kwargs.pop('cache_config', None)
        super().__init__(*args, **kwargs)
        
        self.logger = logging.getLogger(__name__)
        self.middleware_manager = MiddlewareManager()
        self._setup()
        self.add_event_handler("startup", self.on_startup)
        self.add_event_handler("shutdown", self.on_shutdown)
    
    def _setup(self):
        """Set up the application with middleware and routes."""
        # Configure and apply middlewares
        self._setup_middlewares()
        
        # Set up caching
        self._setup_cache()
        
        # Set up admin dashboard
        self._admin = None
        if os.getenv("EAGLE_ADMIN_ENABLED", "true").lower() == "true":
            self.enable_admin()
    
    def _setup_middlewares(self):
        """Configure and apply middlewares based on configuration."""
        config = self.middleware_config
        
        # Configure logging middleware
        logging_config = config.get('logging', {})
        if logging_config.get('enabled', True):
            self.middleware_manager.configure_logging(
                enabled=True,
                log_methods=logging_config.get('methods', ['POST', 'PUT', 'DELETE']),
                excluded_paths=logging_config.get('excluded_paths', [
                    '/health', '/metrics', '/docs', '/redoc', '/openapi.json'
                ]),
                log_request_body=logging_config.get('log_request_body', True),
                log_response_body=logging_config.get('log_response_body', False),
                max_body_size=logging_config.get('max_body_size', 10 * 1024),
                batch_logging=logging_config.get('batch_logging', False)
            )
        
        # Configure CORS middleware
        cors_config = config.get('cors', {})
        if cors_config.get('enabled', True):
            self.middleware_manager.configure_cors(
                enabled=True,
                allow_origins=cors_config.get('origins', ["*"]),
                allow_methods=cors_config.get('methods', ["*"]),
                allow_credentials=cors_config.get('credentials', True)
            )
        
        # Configure rate limiting if enabled
        rate_limit_config = config.get('rate_limit', {})
        if rate_limit_config.get('enabled', False):
            self.middleware_manager.configure_rate_limit(
                enabled=True,
                calls=rate_limit_config.get('calls', 100),
                period=rate_limit_config.get('period', 60)
            )
        
        # Apply all middlewares to the app
        try:
            self.middleware_manager.apply_to_app(self)
            self.logger.info("All middlewares applied successfully")
        except Exception as e:
            self.logger.error(f"Error applying middlewares: {e}")
            raise

def create_app(
    title: str = "Eagle Framework",
    description: str = "A modern Python web framework built on FastAPI",
    version: str = "0.1.0",
    debug: bool = False,
    cache_config: Optional[Dict[str, Any]] = None,
    middleware_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> EagleAPI:
    """
    Create and configure the Eagle application.
    
    Args:
        middleware_config: Middleware configuration dictionary. Example:
            {
                "logging": {
                    "enabled": True,
                    "methods": ["POST", "PUT", "DELETE"],
                    "excluded_paths": ["/health", "/metrics"],
                    "log_request_body": True,
                    "log_response_body": False,
                    "batch_logging": False
                },
                "cors": {
                    "enabled": True,
                    "origins": ["*"],
                    "methods": ["*"],
                    "credentials": True
                },
                "rate_limit": {
                    "enabled": False,
                    "calls": 100,
                    "period": 60
                }
            }
    """
    # Set default middleware config
    default_middleware_config = {
        "logging": {"enabled": True},
        "cors": {"enabled": True},
        "rate_limit": {"enabled": False}
    }
    
    if middleware_config:
        # Merge with defaults
        for key, value in middleware_config.items():
            if key in default_middleware_config:
                default_middleware_config[key].update(value)
            else:
                default_middleware_config[key] = value
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Creating {title} application")
        
        app = EagleAPI(
            title=title,
            description=description,
            version=version,
            debug=debug,
            cache_config=cache_config,
            middleware_config=default_middleware_config,
            **kwargs
        )
        
        # Include routers
        try:
            from .auth import router as auth_router
            app.include_router(auth_router)
            logger.info("Auth router included")
        except ImportError as e:
            logger.warning(f"Could not import auth router: {e}")
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return {"status": "ok", "middleware": "active"}
        
        logger.info("Application initialization complete")
        return app
        
    except Exception as e:
        logger.critical(f"Failed to create application: {e}")
        raise

# Global middleware manager instance
