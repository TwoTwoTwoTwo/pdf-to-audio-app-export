#!/usr/bin/env python3
"""
Quick start script for PDF to Audio App
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("PDF TO AUDIO APP - QUICK START")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Try: pip install -r requirements.txt")
        sys.exit(1)
    
    # Start the app
    print("\nStarting the app...")
    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Try: streamlit run app.py")

if __name__ == "__main__":
    main()
