import subprocess
import os
import sys

def run_dashboard():
    visualizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizer.py"))
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", visualizer_path,
        "--server.headless", "false"
    ])
