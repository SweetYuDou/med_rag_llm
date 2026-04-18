import os

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(
        host=os.getenv('MED_HOST', '0.0.0.0'),
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
