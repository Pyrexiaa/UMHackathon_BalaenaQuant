import subprocess
import os

def run_dashboard():
    filepath = os.path.join(os.path.dirname(__file__), "visualizer.py")
    subprocess.run(["streamlit", "run", filepath])