import os
from pathlib import Path

from flask import Flask, abort, send_from_directory

def create_app():
    project_root = Path(__file__).resolve().parents[2]
    frontend_dist = project_root / 'frontend' / 'dist'
    serve_frontend_dist = os.getenv('MED_SERVE_FRONTEND_DIST', '0') == '1'

    app = Flask(
        __name__,
        static_folder=str(frontend_dist) if serve_frontend_dist and frontend_dist.exists() else None,
        static_url_path='',
    )
    app.json.ensure_ascii = False
    app.json.sort_keys = False
    
    # Manual CORS for demonstration if flask-cors is missing
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    # Register Blueprints
    from .routes import main_bp
    app.register_blueprint(main_bp)

    if serve_frontend_dist and frontend_dist.exists():
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_frontend(path):
            if path.startswith('api/'):
                abort(404)
            target = frontend_dist / path
            if path and target.exists() and target.is_file():
                return send_from_directory(frontend_dist, path)
            return send_from_directory(frontend_dist, 'index.html')
    
    return app
