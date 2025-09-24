#!/bin/bash

# Build script for MuJoCo WASM React App
# Replicates the multi-stage Docker build process
# Usage: ./build.sh [MODEL_PATH]
# Example: ./build.sh /home/user/reachy_mini/descriptions/reachy_mini/mjcf

set -e  # Exit on any error

# Check if model path is provided
if [[ -z "$1" ]]; then
    echo "❌ Error: Model path is required"
    echo "Usage: ./build.sh MODEL_PATH"
    echo "Example: ./build.sh /home/cdussieux/dev/reachy_mini/reachy_mini/src/reachy_mini/descriptions/reachy_mini/mjcf"
    exit 1
fi

MODEL_PATH="$1"

# Validate model path exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

echo "🚀 Starting MuJoCo WASM React App build..."
echo "📁 Using model path: $MODEL_PATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
print_step "Checking prerequisites..."

# Check if we're in the right directory
if [[ ! -f "server.js" || ! -d "mujoco_web" ]]; then
    print_error "This script must be run from the directory containing server.js and mujoco_web submodule"
    exit 1
fi

# Check if required tools are installed
command -v node >/dev/null 2>&1 || { print_error "Node.js is required but not installed. Aborting."; exit 1; }
command -v npm >/dev/null 2>&1 || { print_error "npm is required but not installed. Aborting."; exit 1; }
command -v cmake >/dev/null 2>&1 || { print_error "cmake is required but not installed. Aborting."; exit 1; }
command -v git >/dev/null 2>&1 || { print_error "git is required but not installed. Aborting."; exit 1; }

print_success "Prerequisites check passed"

# Install pnpm if not available
if ! command -v pnpm >/dev/null 2>&1; then
    print_step "Installing pnpm..."
    npm install -g pnpm
    print_success "pnpm installed"
fi

# Check for Emscripten SDK
if [[ ! -d "/emsdk" ]]; then
    print_step "Installing Emscripten SDK..."

    # Clone emsdk if it doesn't exist
    if [[ ! -d "emsdk" ]]; then
        git clone https://github.com/emscripten-core/emsdk.git emsdk
    fi

    cd emsdk
    ./emsdk install latest
    ./emsdk activate latest
    cd ..

    export EMSDK="$(pwd)/emsdk"
    export PATH="$EMSDK:$EMSDK/node/$(ls $EMSDK/node)/bin:$EMSDK/upstream/emscripten:$EMSDK/upstream/bin:$PATH"

    print_success "Emscripten SDK installed and configured"
else
    print_step "Using existing Emscripten SDK..."
    export EMSDK=/emsdk
    export PATH="/emsdk:/emsdk/node/$(ls /emsdk/node)/bin:/emsdk/upstream/emscripten:/emsdk/upstream/bin:$PATH"
fi

# Create workspace directory
WORKSPACE_DIR="build_workspace"
print_step "Setting up workspace: $WORKSPACE_DIR"

rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"

# Copy mujoco_web submodule
print_step "Copying MuJoCo web submodule..."
cp -r mujoco_web "$WORKSPACE_DIR/"
print_success "MuJoCo web submodule copied"

# Copy model files from provided path
print_step "Copying model files from: $MODEL_PATH"
mkdir -p "$WORKSPACE_DIR/models"
cp -r "$MODEL_PATH"/* "$WORKSPACE_DIR/models/"
print_success "Model files copied"

# Build MuJoCo WASM
print_step "Building MuJoCo WASM..."
cd "$WORKSPACE_DIR/mujoco_web"

# Build using emscripten
emcmake cmake -B build -H. -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

print_success "MuJoCo WASM build completed"

# Copy WASM outputs to public directory
print_step "Copying WASM files to public directory..."
mkdir -p public
cp src/wasm/mujoco_wasm.wasm public/mujoco.wasm
cp src/wasm/mujoco_wasm.js public/mujoco.js

print_success "WASM files copied"

# Copy model files to public directory
print_step "Copying models to public directory..."
mkdir -p public/models/reachy
cp -r ../models/* public/models/reachy/

print_success "Models copied to public directory"

# Build React app
print_step "Installing React dependencies..."
pnpm install

print_success "Dependencies installed"

# Backup WASM files for light builds (save to main directory)
print_step "Backing up WASM files for light builds..."
mkdir -p ../../mujoco_web/wasm_backup
cp src/wasm/mujoco_wasm.wasm ../../mujoco_web/wasm_backup/
cp src/wasm/mujoco_wasm.js ../../mujoco_web/wasm_backup/
print_success "WASM files backed up to main directory"

print_step "Patching mujoco_wasm.js for Vite..."
sed -i 's/, {/, \/* @vite-ignore *\/ {/' src/wasm/mujoco_wasm.js

print_step "Building React application..."
pnpm run build

print_success "React build completed"

# Go back to original directory
cd ../..

# Set up production files
print_step "Setting up production files..."
rm -rf dist
mkdir -p dist

# Copy built React app
cp -r "$WORKSPACE_DIR/mujoco_web/dist"/* dist/

# Copy model scenes to the correct location
print_step "Copying model scenes to dist..."
mkdir -p dist/examples/scenes/reachy
cp -r "$WORKSPACE_DIR/models"/* dist/examples/scenes/reachy/

print_success "Production files ready"

# Create or update package.json for production server
print_step "Creating production package.json..."
cat > package.json << EOF
{
  "name": "mujoco_web_server",
  "version": "1.0.0",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "build": "./build.sh"
  },
  "dependencies": {
    "express": "^4.18.2"
  }
}
EOF

# Install production dependencies
if [[ ! -d "node_modules" ]]; then
    print_step "Installing production dependencies..."
    npm install --production
    print_success "Production dependencies installed"
fi

# Cleanup workspace
print_step "Cleaning up workspace..."
rm -rf "$WORKSPACE_DIR"
print_success "Workspace cleaned up"

print_success "🎉 Build completed successfully!"
echo ""
echo -e "${GREEN}📁 Built files are in: ./dist${NC}"
echo -e "${GREEN}🚀 To start the server: npm start${NC}"
echo -e "${GREEN}🌐 Server will run on: http://localhost:7860${NC}"