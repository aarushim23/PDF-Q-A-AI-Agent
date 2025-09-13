#!/usr/bin/env python3
"""
Installation script to fix missing dependencies
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_package(package):
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error installing {package}: {e}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is missing")
        return False

def main():
    """Main installation function."""
    print("🔧 Dependency Installer for Document Q&A System")
    print("=" * 55)
    
    # Critical packages that are often missing
    critical_packages = [
        ("sentencepiece", "sentencepiece"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("pdfplumber", "pdfplumber"),
        ("arxiv", "arxiv"),
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("requests", "requests"),
        ("Pillow", "PIL"),
        ("numpy", "numpy")
    ]
    
    # Optional packages
    optional_packages = [
        ("camelot-py[cv]", "camelot"),
        ("regex", "regex")
    ]
    
    missing_critical = []
    missing_optional = []
    
    print("🔍 Checking critical dependencies...")
    for package, import_name in critical_packages:
        if not check_package(package, import_name):
            missing_critical.append(package)
    
    print("\n🔍 Checking optional dependencies...")
    for package, import_name in optional_packages:
        if not check_package(package, import_name):
            missing_optional.append(package)
    
    # Install missing critical packages
    if missing_critical:
        print(f"\n📦 Installing {len(missing_critical)} critical packages...")
        failed_installations = []
        
        for package in missing_critical:
            if not install_package(package):
                failed_installations.append(package)
        
        if failed_installations:
            print(f"\n❌ Failed to install: {', '.join(failed_installations)}")
            print("💡 Try installing manually with:")
            for package in failed_installations:
                print(f"   pip install {package}")
        else:
            print("\n✅ All critical packages installed successfully!")
    else:
        print("\n✅ All critical dependencies are satisfied!")
    
    # Install missing optional packages
    if missing_optional:
        print(f"\n📦 Installing {len(missing_optional)} optional packages...")
        for package in missing_optional:
            install_package(package)  # Don't fail if optional packages fail
    
    print("\n" + "=" * 55)
    print("🎉 Installation complete!")
    
    # Verify key packages for the LLM issue
    print("\n🧪 Testing key imports...")
    test_imports = [
        ("sentencepiece", "✅ T5 tokenizer will work"),
        ("transformers", "✅ Hugging Face models available"),
        ("torch", "✅ PyTorch available"),
        ("streamlit", "✅ Web app framework ready")
    ]
    
    all_good = True
    for package, success_msg in test_imports:
        try:
            __import__(package)
            print(success_msg)
        except ImportError:
            print(f"❌ {package} still not available")
            all_good = False
    
    if all_good:
        print("\n🚀 System is ready! You can now run:")
        print("   python debug_tool.py")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Some packages are still missing. Please install them manually.")
        print("💡 Common solutions:")
        print("   1. Restart your terminal/command prompt")
        print("   2. Try: pip install --upgrade pip")
        print("   3. Try: pip install sentencepiece transformers torch")

if __name__ == "__main__":
    main()