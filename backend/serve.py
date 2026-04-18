import argparse
import os

from app import create_app


app = create_app()


def parse_args():
    parser = argparse.ArgumentParser(description='Run the backend as a long-lived service.')
    parser.add_argument('--host', default=os.getenv('MED_BACKEND_HOST', '0.0.0.0'))
    parser.add_argument('--port', type=int, default=int(os.getenv('MED_BACKEND_PORT', '5000')))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)
