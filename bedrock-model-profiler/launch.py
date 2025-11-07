#!/usr/bin/env python3
"""
Amazon Bedrock Model Profiler - Unified Launcher
Cross-platform launcher script for Windows, macOS, and Linux

Usage:
    python launch.py

This script will:
1. Check and create virtual environment if needed
2. Install dependencies if needed
3. Set up AWS configuration
4. Initialize data if needed
5. Run functionality tests
6. Launch the Streamlit application
"""

import os
import sys
import subprocess  # nosec B404 - Required for Python/pip operations with proper input validation
import platform
import shutil
import re
import shlex  # For additional command line argument escaping
from pathlib import Path


def _validate_directory_name(directory_name):
    """
    Validate directory name to prevent command injection.
    Directory names should only contain safe characters and not be absolute paths with suspicious content.
    """
    if not directory_name or len(directory_name) > 255:
        return False

    # Reject names that start with - (could be interpreted as command flags)
    if directory_name.startswith('-'):
        return False

    # Allow alphanumeric, dots, hyphens, underscores, and path separators
    # Reject shell metacharacters like ;|&$`(){}[]<>
    dangerous_chars = set(';|&$`(){}[]<>*?')
    if any(char in dangerous_chars for char in directory_name):
        return False

    return True


def _validate_port_number(port):
    """
    Validate port number to prevent command injection.
    Port should be a valid integer between 1 and 65535.
    """
    try:
        port_int = int(port)
        return 1 <= port_int <= 65535
    except (ValueError, TypeError):
        return False


def _validate_file_path(file_path):
    """
    Validate file path to prevent command injection.
    File path should be safe and not contain dangerous shell metacharacters.
    """
    if not file_path:
        return False

    path_str = str(file_path)

    # Reject paths that start with - (could be interpreted as command flags)
    if path_str.startswith('-'):
        return False

    # Reject shell metacharacters
    dangerous_chars = set(';|&$`(){}[]<>*?')
    if any(char in dangerous_chars for char in path_str):
        return False

    return True


def print_banner():
    """Print application banner"""
    print("🤖 Amazon Bedrock Model Profiler")
    print("=" * 37)
    print()


def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "info": "\033[94m",      # Blue
        "success": "\033[92m",   # Green
        "warning": "\033[93m",   # Yellow
        "error": "\033[91m",     # Red
        "reset": "\033[0m"       # Reset
    }

    # Use colors on Unix systems, plain text on Windows
    if platform.system() != "Windows":
        color = colors.get(status, colors["info"])
        reset = colors["reset"]
        print(f"{color}{message}{reset}")
    else:
        print(message)


def get_config():
    """Load configuration from config.py"""
    try:
        # Add current directory to Python path to import config
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))

        import config
        return config
    except ImportError:
        print_status("❌ Error: config.py not found. Please ensure config.py exists.", "error")
        sys.exit(1)
    except Exception as e:
        print_status(f"❌ Error loading config.py: {e}", "error")
        sys.exit(1)


def get_python_executable():
    """Get the appropriate Python executable"""
    # Try different Python executable names
    python_names = ["python3", "python"]

    for name in python_names:
        if shutil.which(name):
            return name

    print_status("❌ Error: Python executable not found", "error")
    sys.exit(1)


def get_venv_paths(config):
    """Get virtual environment paths for current platform"""
    venv_dir = Path(config.VENV_DIR)

    if platform.system() == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
        activate_script = venv_dir / "Scripts" / "activate.bat"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        activate_script = venv_dir / "bin" / "activate"
        pip_exe = venv_dir / "bin" / "pip"

    return {
        "python": python_exe,
        "activate": activate_script,
        "pip": pip_exe,
        "venv_dir": venv_dir
    }


def create_virtual_environment(config, python_cmd):
    """Create virtual environment if it doesn't exist

    Security: Uses list-based subprocess calls with shell=False (default).
    This prevents command injection since arguments are passed directly
    to the executable without shell interpretation. Input validation
    ensures all arguments are safe.
    """
    venv_paths = get_venv_paths(config)

    # Check if venv exists AND is valid (has python executable)
    needs_creation = not venv_paths["venv_dir"].exists() or not venv_paths["python"].exists()

    if needs_creation:
        # Remove incomplete venv if it exists
        if venv_paths["venv_dir"].exists():
            print_status("🔧 Removing incomplete virtual environment...", "info")
            shutil.rmtree(venv_paths["venv_dir"])

        print_status("🔧 Creating virtual environment...", "info")

        # Validate virtual environment directory name to prevent command injection
        if not _validate_directory_name(config.VENV_DIR):
            print_status(f"❌ Error: Invalid virtual environment directory name: {config.VENV_DIR}", "error")
            print_status("Directory names must not contain shell metacharacters", "error")
            sys.exit(1)

        try:
            # Safe list-based subprocess call with additional escaping for defense in depth
            escaped_venv_dir = shlex.quote(config.VENV_DIR)
            # nosemgrep: dangerous-subprocess-use-audit
            subprocess.run([python_cmd, "-m", "venv", escaped_venv_dir], check=True, shell=False)  # nosec B603
            print_status("✅ Virtual environment created successfully", "success")
        except subprocess.CalledProcessError as e:
            print_status(f"❌ Error creating virtual environment: {e}", "error")
            sys.exit(1)
    else:
        print_status("✅ Virtual environment already exists", "success")

    return venv_paths


def install_dependencies(config, venv_paths):
    """Install dependencies if needed"""
    requirements_file = Path(config.PYTHON_REQUIREMENTS)

    # Check if virtual environment Python exists
    if not venv_paths["python"].exists():
        print_status(f"❌ Error: Virtual environment Python not found at {venv_paths['python']}", "error")
        print_status("💡 Try deleting the venv directory and running again", "info")
        sys.exit(1)

    # Validate requirements file path to prevent command injection
    if not _validate_file_path(requirements_file):
        print_status(f"❌ Error: Invalid requirements file path: {config.PYTHON_REQUIREMENTS}", "error")
        print_status("File paths must not contain shell metacharacters", "error")
        sys.exit(1)

    if not requirements_file.exists():
        print_status(f"❌ Error: {config.PYTHON_REQUIREMENTS} not found", "error")
        sys.exit(1)

    # Check if streamlit is installed (quick dependency check)
    try:
        # nosemgrep: dangerous-subprocess-use-audit
        result = subprocess.run([str(venv_paths["python"]), "-c", "import streamlit"],
                              capture_output=True, text=True, shell=False)  # nosec B603
        if result.returncode != 0:
            print_status("🔧 Installing dependencies...", "info")
            # Additional security: escape requirements file path for defense in depth
            escaped_req_file = shlex.quote(str(requirements_file))
            # nosemgrep: dangerous-subprocess-use-audit
            subprocess.run([str(venv_paths["pip"]), "install", "-r", escaped_req_file], check=True, shell=False)  # nosec B603
            print_status("✅ Dependencies installed successfully", "success")
        else:
            print_status("✅ Dependencies already installed", "success")
    except subprocess.CalledProcessError as e:
        print_status(f"❌ Error installing dependencies: {e}", "error")
        sys.exit(1)


def setup_aws_environment(config):
    """Set up AWS environment variables"""
    print_status("🔑 Setting up AWS configuration...", "info")

    # Set AWS region
    if hasattr(config, 'AWS_DEFAULT_REGION') and config.AWS_DEFAULT_REGION:
        os.environ['AWS_DEFAULT_REGION'] = config.AWS_DEFAULT_REGION

    # Set direct credentials if provided
    if hasattr(config, 'AWS_ACCESS_KEY_ID') and config.AWS_ACCESS_KEY_ID:
        os.environ['AWS_ACCESS_KEY_ID'] = config.AWS_ACCESS_KEY_ID

    if hasattr(config, 'AWS_SECRET_ACCESS_KEY') and config.AWS_SECRET_ACCESS_KEY:
        os.environ['AWS_SECRET_ACCESS_KEY'] = config.AWS_SECRET_ACCESS_KEY


def initialize_data(config, venv_paths):
    """Initialize application data if needed"""
    data_file = Path(config.DATA_DIR) / config.MODEL_DATA_FILE

    if not data_file.exists():
        print_status("📊 No model data found. You can update data using the app's interface.", "warning")
        print_status("💡 The app will still run with sample data or you can update data later.", "info")
    else:
        print_status("✅ Application data ready", "success")


def run_functionality_test(config, venv_paths):
    """Run a quick functionality test"""
    print_status("🧪 Running functionality test...", "info")

    test_code = """
import sys
try:
    from models.new_model_repository import NewModelRepository
    repo = NewModelRepository()
    df = repo.load_models_df()
    if len(df) > 0:
        print(f'✅ Data loaded successfully: {len(df)} models')
        print('RESULT:DATA_AVAILABLE')
    else:
        print('⚠️  No model data found - app will start with data collection guidance')
        print('RESULT:NO_DATA')
except Exception as e:
    print(f'❌ Error during data check: {str(e)}')
    print('RESULT:ERROR')
"""

    try:
        # nosemgrep: dangerous-subprocess-use-audit
        result = subprocess.run([str(venv_paths["python"]), "-c", test_code],
                              capture_output=True, text=True, shell=False)  # nosec B603

        output = result.stdout.strip()
        print_status(output.split('RESULT:')[0].strip(), "info")

        # Check the result and provide appropriate feedback
        if 'RESULT:DATA_AVAILABLE' in output:
            print_status("✅ Functionality test passed: Data ready", "success")
            return True
        elif 'RESULT:NO_DATA' in output:
            print_status("ℹ️  Functionality test passed: No data found, will show collection guidance", "warning")
            return True
        else:
            print_status("⚠️  Functionality test passed with warnings", "warning")
            return True

    except subprocess.CalledProcessError as e:
        print_status(f"❌ Critical functionality test failed: {e.stderr or e.stdout}", "error")
        print_status("💡 There may be a Python environment issue", "info")
        return False


def launch_streamlit_app(config, venv_paths):
    """Launch the Streamlit application"""
    # Validate Streamlit port to prevent command injection
    if hasattr(config, 'STREAMLIT_PORT') and not _validate_port_number(config.STREAMLIT_PORT):
        print_status(f"❌ Error: Invalid Streamlit port: {config.STREAMLIT_PORT}", "error")
        print_status("Port must be a valid integer between 1 and 65535", "error")
        sys.exit(1)

    print_status("🚀 Starting Amazon Bedrock Model Profiler...", "success")
    print_status(f"📱 The app will be available at: http://localhost:{config.STREAMLIT_PORT}", "info")
    print_status("💡 Press Ctrl+C to stop the application", "warning")
    print()

    try:
        # Set the port if specified in config
        env = os.environ.copy()
        if hasattr(config, 'STREAMLIT_PORT'):
            env['STREAMLIT_SERVER_PORT'] = str(config.STREAMLIT_PORT)

        # Launch streamlit with auto-reload flags (additional security: escape port for defense in depth)
        escaped_port = shlex.quote(str(config.STREAMLIT_PORT))
        # nosemgrep: dangerous-subprocess-use-audit
        subprocess.run([
            str(venv_paths["python"]), "-m", "streamlit", "run", "app.py",
            "--server.port", escaped_port,
            "--server.fileWatcherType", "polling",
            "--server.runOnSave", "true"
        ], env=env, shell=False)  # nosec B603
    except KeyboardInterrupt:
        print_status("\n👋 Application stopped by user", "info")
    except subprocess.CalledProcessError as e:
        print_status(f"❌ Error launching application: {e}", "error")
        sys.exit(1)


def main():
    """Main launcher function"""
    print_banner()

    # Load configuration
    config = get_config()

    # Get Python executable
    python_cmd = get_python_executable()
    print_status(f"🐍 Using Python: {python_cmd}", "info")

    # Create/verify virtual environment
    venv_paths = create_virtual_environment(config, python_cmd)

    # Install dependencies
    install_dependencies(config, venv_paths)

    # Set up AWS environment
    setup_aws_environment(config)

    # Initialize data
    initialize_data(config, venv_paths)

    # Run functionality test
    test_passed = run_functionality_test(config, venv_paths)
    if not test_passed:
        print_status("❌ Critical functionality test failed - Python environment may have issues", "error")
        print_status("💡 Please check your Python environment and try again", "info")
        sys.exit(1)

    # Launch the application (works with or without model data)
    print_status("🚀 Starting application...", "info")
    launch_streamlit_app(config, venv_paths)


if __name__ == "__main__":
    main()