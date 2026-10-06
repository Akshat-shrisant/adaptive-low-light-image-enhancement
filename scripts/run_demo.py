"""
Launch Script for the Faculty Demonstration Web Interface
Course: Image and Video Processing (IVP) Mini-Project

Usage:
    python scripts/run_demo.py [--port 5000]

Launches a local web demonstration server at:
    http://127.0.0.1:5000
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.app import app


def main():
    parser = argparse.ArgumentParser(description="Run Adaptive Low-Light Enhancement Demonstration")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the demonstration server on (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    print("=" * 75)
    print("ADAPTIVE LOW-LIGHT ENHANCEMENT — FACULTY DEMONSTRATION")
    print("Course: B.Tech Image and Video Processing (IVP) Mini-Project")
    print("=" * 75)
    print(f"-> Local Server running at: http://{args.host}:{args.port}")
    print("-> Open the URL in your web browser to interact with the demo.")
    print("-> Press Ctrl+C to terminate the server.")
    print("=" * 75)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
