#!/usr/bin/env python3
"""
Data2Pydantic Mapping Tool - Setup Script
Automatically sets up virtual environment and installs dependencies
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.OKBLUE):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.ENDC}")

def print_header(message):
    """Print a formatted header"""
    print()
    print_colored("=" * 60, Colors.HEADER)
    print_colored(f"  {message}", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 60, Colors.HEADER)
    print()

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}.{version.micro}", Colors.FAIL)
        print_colored("Please install Python 3.8 or higher from https://python.org", Colors.WARNING)
        sys.exit(1)
    print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} detected", Colors.OKGREEN)
    return True

def get_venv_path():
    """Get the virtual environment path"""
    return Path.cwd() / "venv"

def get_activation_command():
    """Get the appropriate activation command for the current OS"""
    system = platform.system()
    venv_path = get_venv_path()
    
    if system == "Windows":
        activate_path = venv_path / "Scripts" / "activate.bat"
        activate_cmd = str(activate_path)
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Unix-like (Linux, macOS)
        activate_path = venv_path / "bin" / "activate"
        activate_cmd = f"source {activate_path}"
        python_path = venv_path / "bin" / "python"
    
    return activate_cmd, python_path

def create_venv():
    """Create virtual environment"""
    venv_path = get_venv_path()
    
    # Check if venv already exists
    if venv_path.exists():
        print_colored(f"ℹ️  Virtual environment already exists at {venv_path}", Colors.WARNING)
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response == 'y':
            print_colored("Removing existing virtual environment...", Colors.OKCYAN)
            shutil.rmtree(venv_path)
        else:
            return True
    
    print_colored("Creating virtual environment...", Colors.OKCYAN)
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print_colored(f"✅ Virtual environment created at {venv_path}", Colors.OKGREEN)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to create virtual environment: {e}", Colors.FAIL)
        return False

def install_dependencies():
    """Install required dependencies"""
    _, python_path = get_activation_command()
    
    print_colored("Installing dependencies...", Colors.OKCYAN)
    
    # Upgrade pip first
    print_colored("  Upgrading pip...", Colors.OKCYAN)
    try:
        subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print_colored(f"  ⚠️  Warning: Failed to upgrade pip: {e}", Colors.WARNING)
    
    # Install requirements
    requirements_file = Path.cwd() / "requirements.txt"
    if not requirements_file.exists():
        print_colored("❌ requirements.txt not found!", Colors.FAIL)
        return False
    
    print_colored("  Installing packages from requirements.txt...", Colors.OKCYAN)
    try:
        result = subprocess.run(
            [str(python_path), "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_colored("✅ All dependencies installed successfully", Colors.OKGREEN)
            return True
        else:
            print_colored(f"❌ Failed to install dependencies", Colors.FAIL)
            print_colored(f"Error: {result.stderr}", Colors.FAIL)
            return False
    except Exception as e:
        print_colored(f"❌ Error installing dependencies: {e}", Colors.FAIL)
        return False

def create_run_scripts():
    """Create platform-specific run scripts"""
    venv_path = get_venv_path()
    system = platform.system()
    
    print_colored("Creating run scripts...", Colors.OKCYAN)
    
    # Windows batch script only
    if system == "Windows":
        script_content = f"""@echo off
echo Starting Data2Pydantic Mapping Tool...
call "{venv_path}\\Scripts\\activate.bat"
python -m streamlit run app.py
pause
"""
        script_path = Path.cwd() / "run.bat"
        script_path.write_text(script_content)
        print_colored(f"✅ Created run.bat", Colors.OKGREEN)
    else:
        print_colored("ℹ️ No run script created for Unix/Mac (use manual activation)", Colors.OKCYAN)

def create_env_file():
    """Create .env file from .example.env if it doesn't exist"""
    env_file = Path.cwd() / ".env"
    example_env = Path.cwd() / ".example.env"
    
    if not env_file.exists() and example_env.exists():
        print_colored("Creating .env file from .example.env...", Colors.OKCYAN)
        shutil.copy(example_env, env_file)
        print_colored("✅ Created .env file (please update with your API keys)", Colors.OKGREEN)
        return True
    elif env_file.exists():
        print_colored("ℹ️  .env file already exists", Colors.WARNING)
        return True
    return False

def test_installation():
    """Test if the installation works"""
    _, python_path = get_activation_command()
    
    print_colored("Testing installation...", Colors.OKCYAN)
    
    # Test imports
    test_script = """
import streamlit
import pandas
import pydantic
import openpyxl
import xlsxwriter
import openai
print("All imports successful!")
"""
    
    try:
        result = subprocess.run(
            [str(python_path), "-c", test_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_colored("✅ Installation test passed", Colors.OKGREEN)
            return True
        else:
            print_colored("❌ Installation test failed", Colors.FAIL)
            print_colored(f"Error: {result.stderr}", Colors.FAIL)
            return False
    except Exception as e:
        print_colored(f"❌ Test failed: {e}", Colors.FAIL)
        return False

def print_next_steps():
    """Print instructions for next steps"""
    system = platform.system()
    activate_cmd, _ = get_activation_command()
    
    print_header("🎉 Setup Complete!")
    
    print_colored("Next steps:", Colors.BOLD)
    print()
    
    if system == "Windows":
        print_colored("Option 1 - Use the run script:", Colors.OKCYAN)
        print("  Double-click 'run.bat' or run:")
        print("  > .\\run.bat")
        print()
        print_colored("Option 2 - Manual activation:", Colors.OKCYAN)
        print(f"  > {activate_cmd}")
        print("  > streamlit run app.py")
    else:
        print_colored("Manual activation:", Colors.OKCYAN)
        print(f"  $ {activate_cmd}")
        print("  $ streamlit run app.py")
    
    print()
    print_colored("📝 Don't forget to:", Colors.WARNING)
    print("  1. Update .env with your API keys (for LLM features)")
    print("  2. The app will open at http://localhost:8501")
    print()
    print_colored("For more information, see README.md", Colors.OKCYAN)

def main():
    """Main setup process"""
    print_header("Data2Pydantic Mapping Tool - Setup")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Detect OS
    system = platform.system()
    print_colored(f"Operating System: {system}", Colors.OKBLUE)
    
    # Create virtual environment
    if not create_venv():
        print_colored("Setup failed at virtual environment creation", Colors.FAIL)
        return
    
    # Install dependencies
    if not install_dependencies():
        print_colored("Setup failed at dependency installation", Colors.FAIL)
        print_colored("Try running the setup again or install manually", Colors.WARNING)
        return
    
    # Create run scripts
    create_run_scripts()
    
    # Create .env file
    create_env_file()
    
    # Test installation
    test_installation()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_colored("Setup cancelled by user", Colors.WARNING)
        sys.exit(0)
    except Exception as e:
        print_colored(f"Unexpected error: {e}", Colors.FAIL)
        sys.exit(1)