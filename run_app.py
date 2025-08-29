#!/usr/bin/env python3
"""
Launcher script for the Econometrics Toolkit Streamlit app.
This script ensures the Python path is set correctly before launching Streamlit.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Add the script directory to Python path
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    # Verify we can import the analysis_functions
    try:
        from analysis_functions import load_functions
        funcs = load_functions()
        print(f"✅ Successfully imported analysis_functions. Found {len(funcs)} functions:")
        for func in funcs:
            print(f"   - {func.display_name}")
    except ImportError as e:
        print(f"❌ Failed to import analysis_functions: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}...")
        return 1
    
    # Launch Streamlit
    print("🚀 Launching Streamlit app...")
    streamlit_script = script_dir / "app" / "streamlit_app.py"
    
    if not streamlit_script.exists():
        print(f"❌ Streamlit script not found at: {streamlit_script}")
        return 1
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_script)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit failed to run: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Streamlit stopped by user")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
