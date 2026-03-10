#!/bin/bash

# SpectraQuant-AI Installation Script
# Automated installation for Linux and macOS
# For Windows, see INSTALLATION.md

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main installation
main() {
    print_header "SpectraQuant-AI Installation"
    echo ""
    
    # Check prerequisites
    print_info "Checking prerequisites..."
    echo ""
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.11+ required, found Python $PYTHON_VERSION"
            echo "Please install Python 3.11 or higher and try again."
            exit 1
        fi
    else
        print_error "Python 3 not found"
        echo "Please install Python 3.11+ and try again."
        exit 1
    fi
    
    # Check Git
    if command_exists git; then
        print_success "Git found"
    else
        print_error "Git not found"
        echo "Please install Git and try again."
        exit 1
    fi
    
    # Check pip
    if command_exists pip3; then
        print_success "pip found"
    else
        print_warning "pip not found, will try to install"
    fi
    
    echo ""
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_info "Detected Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_info "Detected macOS"
    else
        OS="unknown"
        print_warning "Unknown OS: $OSTYPE"
    fi
    
    echo ""
    
    # Ask installation type
    echo "Select installation type:"
    echo "1) Core only (basic features)"
    echo "2) Full (core + intelligence layer)"
    read -p "Enter choice [1-2]: " INSTALL_TYPE
    
    if [ "$INSTALL_TYPE" != "1" ] && [ "$INSTALL_TYPE" != "2" ]; then
        print_error "Invalid choice"
        exit 1
    fi
    
    echo ""
    print_header "Starting Installation"
    echo ""
    
    # Create virtual environment
    print_info "Creating virtual environment..."
    if python3 -m venv .venv; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source .venv/bin/activate
    if [ $? -eq 0 ]; then
        print_success "Virtual environment activated"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
    
    # Upgrade pip
    print_info "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_success "pip upgraded"
    else
        print_warning "Failed to upgrade pip (continuing anyway)"
    fi
    
    # Install core dependencies
    print_info "Installing core dependencies (this may take 5-15 minutes)..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_success "Core dependencies installed"
    else
        print_error "Failed to install core dependencies"
        exit 1
    fi
    
    # Install intelligence layer if requested
    if [ "$INSTALL_TYPE" = "2" ]; then
        print_info "Installing intelligence layer dependencies..."
        pip install -r trading_assistant/requirements.txt
        if [ $? -eq 0 ]; then
            print_success "Intelligence layer dependencies installed"
        else
            print_error "Failed to install intelligence layer dependencies"
            exit 1
        fi
    fi
    
    # Install package
    print_info "Installing SpectraQuant package..."
    pip install -e .
    if [ $? -eq 0 ]; then
        print_success "SpectraQuant package installed"
    else
        print_error "Failed to install SpectraQuant package"
        exit 1
    fi
    
    echo ""
    print_header "Verifying Installation"
    echo ""
    
    # Verify CLI
    print_info "Checking CLI..."
    if command_exists spectraquant; then
        print_success "CLI is available"
    else
        print_warning "CLI not found in PATH (may need to restart shell)"
    fi
    
    # Run doctor
    print_info "Running environment diagnostics..."
    spectraquant doctor
    
    echo ""
    print_header "Installation Complete!"
    echo ""
    
    print_success "SpectraQuant-AI has been successfully installed"
    echo ""
    echo "Next steps:"
    echo "  1. Activate virtual environment: source .venv/bin/activate"
    echo "  2. View help: spectraquant --help"
    echo "  3. Configure: edit config.yaml"
    echo "  4. Download data: spectraquant download"
    echo "  5. Run pipeline: spectraquant refresh"
    echo ""
    
    if [ "$INSTALL_TYPE" = "2" ]; then
        echo "Intelligence Layer installed. To setup:"
        echo "  python scripts/bootstrap_intelligence.py"
        echo ""
    fi
    
    echo "For more information, see:"
    echo "  - README.md - Main documentation"
    echo "  - INSTALLATION.md - Detailed installation guide"
    echo "  - DEPENDENCIES.md - Dependency reference"
    echo ""
    
    print_success "Happy trading! 📈🚀"
}

# Run main
main
