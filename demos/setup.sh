#!/bin/bash
# AI Learning Path - Demo Setup Script
# This script sets up the environment for running all demo scripts

set -e  # Exit on any error

echo "🚀 Setting up AI Learning Path Demo Environment"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d" " -f2 | cut -d"." -f1,2)
print_success "Python $PYTHON_VERSION found"

# Check if pip is installed
print_status "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip."
    exit 1
fi
print_success "pip3 found"

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "ai-learning-env" ]; then
    python3 -m venv ai-learning-env
    print_success "Virtual environment created: ai-learning-env"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source ai-learning-env/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip
print_success "pip upgraded"

# Install dependencies
print_status "Installing demo dependencies..."
if [ -f "demos/requirements.txt" ]; then
    pip install -r demos/requirements.txt
    print_success "All dependencies installed"
else
    print_error "requirements.txt not found in demos/ directory"
    exit 1
fi

# Check environment variables
print_status "Checking environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
    print_warning "OPENAI_API_KEY is not set"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-openai-api-key'"
else
    print_success "OPENAI_API_KEY is set"
fi

if [ -z "$SERPER_API_KEY" ]; then
    print_warning "SERPER_API_KEY is not set (optional for search functionality)"
    echo "To enable search features, set:"
    echo "export SERPER_API_KEY='your-serper-api-key'"
else
    print_success "SERPER_API_KEY is set"
fi

# Create demo runner script
print_status "Creating demo runner script..."
cat > run_demo.sh << 'EOF'
#!/bin/bash
# Demo Runner Script

source ai-learning-env/bin/activate

echo "Available demos:"
echo "1. PydanticAI Demo"
echo "2. LangChain Demo"
echo "3. OpenAI Demo"
echo "4. Ango AI Demo"
echo "5. CrewAI Demo"
echo "6. Run All Demos"
echo"
read -p "Choose a demo (1-6): " choice

case $choice in
    1)
        echo "Running PydanticAI Demo..."
        python demos/pydantic_ai_demo.py
        ;;
    2)
        echo "Running LangChain Demo..."
        python demos/langchain_demo.py
        ;;
    3)
        echo "Running OpenAI Demo..."
        python demos/openai_demo.py
        ;;
    4)
        echo "Running Ango AI Demo..."
        python demos/agno_demo.py
        ;;
    5)
        echo "Running CrewAI Demo..."
        python demos/crewai_demo.py
        ;;
    6)
        echo "Running All Demos..."
        echo "Note: This will take several minutes to complete."
        read -p "Continue? (y/N): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            python demos/pydantic_ai_demo.py
            echo "\n" + "=" * 50
            python demos/langchain_demo.py
            echo "\n" + "=" * 50
            python demos/openai_demo.py
            echo "\n" + "=" * 50
            python demos/agno_demo.py
            echo "\n" + "=" * 50
            python demos/crewai_demo.py
        fi
        ;;
    *)
        echo "Invalid choice. Please select 1-6."
        ;;
esac
EOF

chmod +x run_demo.sh
print_success "Demo runner script created: run_demo.sh"

# Final instructions
echo
echo "🎉 Setup Complete!"
echo "=================="
echo
echo "To get started:"
echo "1. Set your API keys (if not already done):"
echo "   export OPENAI_API_KEY='your-openai-api-key'"
echo "   export SERPER_API_KEY='your-serper-api-key'  # Optional"
echo
echo "2. Run demos:"
echo "   ./run_demo.sh  # Interactive demo runner"
echo "   # OR run individual demos:"
echo "   source ai-learning-env/bin/activate"
echo "   python demos/pydantic_ai_demo.py"
echo "   python demos/langchain_demo.py"
echo "   python demos/openai_demo.py"
echo "   python demos/agno_demo.py"
echo "   python demos/crewai_demo.py"
echo
echo "3. Read the documentation:"
echo "   - Main research: AI_Learning_Path_Research.md"
echo "   - Project README: README.md"
echo
echo "Happy learning! 🚀"