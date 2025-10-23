#!/bin/bash

# Light Build script for MuJoCo WASM React App
# Only rebuilds the React app part, skipping MuJoCo WASM compilation

set -e  # Exit on any error

echo "🚀 Starting light build (React app only)..."

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

# Check if model path is provided
if [[ -z "$1" ]]; then
    echo "❌ Error: Model path is required"
    echo "Usage: ./light_build.sh MODEL_PATH"
    echo "Example: ./light_build.sh /home/cdussieux/dev/reachy_mini/reachy_mini/src/reachy_mini/descriptions/reachy_mini/mjcf"
    exit 1
fi

MODEL_PATH="$1"

# Validate model path exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

echo "📁 Using model path: $MODEL_PATH"

# Check if MuJoCo WASM files exist and restore them if needed
print_step "Checking for MuJoCo WASM files..."

# Check if backup files exist (from full build)
if [[ -f "mujoco_web/wasm_backup/mujoco_wasm.wasm" ]] && [[ -f "mujoco_web/wasm_backup/mujoco_wasm.js" ]]; then
    print_step "Restoring WASM files from backup..."

    # Restore original WASM files from backup
    mkdir -p mujoco_web/src/wasm
    cp mujoco_web/wasm_backup/mujoco_wasm.wasm mujoco_web/src/wasm/
    cp mujoco_web/wasm_backup/mujoco_wasm.js mujoco_web/src/wasm/

    print_success "WASM files restored from backup"
# Check if files already exist in source directory
elif [[ -f "mujoco_web/src/wasm/mujoco_wasm.wasm" ]] && [[ -f "mujoco_web/src/wasm/mujoco_wasm.js" ]]; then
    print_success "WASM files found in source directory"
else
    print_error "MuJoCo WASM files not found! Please run full build first: ./build.sh"
    print_error "Checked locations:"
    echo "  - mujoco_web/wasm_backup/"
    echo "  - mujoco_web/src/wasm/"
    exit 1
fi

# Check for pnpm
if ! command -v pnpm >/dev/null 2>&1; then
    print_step "Installing pnpm..."
    npm install -g pnpm
    print_success "pnpm installed"
fi

# Work in the mujoco_web directory
cd mujoco_web

print_step "Updating model files in public directory..."

# Ensure public directory exists
mkdir -p public/models/reachy

# Copy model files to public directory
cp -r "$MODEL_PATH"/* public/models/reachy/
print_success "Model files updated in public directory"

# Copy WASM files to public directory (in case they're missing)
print_step "Ensuring WASM files are in public directory..."
mkdir -p public
cp src/wasm/mujoco_wasm.wasm public/mujoco.wasm
cp src/wasm/mujoco_wasm.js public/mujoco.js
print_success "WASM files ready"

# Install dependencies if node_modules doesn't exist
if [[ ! -d "node_modules" ]]; then
    print_step "Installing React dependencies..."
    pnpm install
    print_success "Dependencies installed"
else
    print_step "Dependencies already installed, skipping..."
fi

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
cp -r mujoco_web/dist/* dist/

# Copy model scenes to the correct location
print_step "Copying model scenes to dist..."
mkdir -p dist/examples/scenes/reachy
cp -r "$MODEL_PATH"/* dist/examples/scenes/reachy/
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
    "build": "./build.sh",
    "light-build": "./light_build.sh"
  },
  "dependencies": {
    "express": "^4.18.2"
  }
}
EOF

# Install production dependencies if needed
if [[ ! -d "node_modules" ]]; then
    print_step "Installing production dependencies..."
    npm install --production
    print_success "Production dependencies installed"
fi

# Create bin directory with essential files for FastAPI app
print_step "Creating bin directory with essential app files..."
rm -rf bin
mkdir -p bin/assets

# Copy essential files for FastAPI integration
cp dist/vite.svg bin/
cp dist/mujoco.js bin/
cp dist/mujoco.wasm bin/

# Create index.html with generic asset references
sed 's|/assets/index-[^.]*\.js|/assets/index.js|g; s|/assets/index-[^.]*\.css|/assets/index.css|g' dist/index.html > bin/index.html
print_success "Created index.html with generic asset references"

# Copy React app assets with generic names
INDEX_JS=$(ls dist/assets/index-*.js 2>/dev/null | head -1)
INDEX_CSS=$(ls dist/assets/index-*.css 2>/dev/null | head -1)

if [[ -n "$INDEX_JS" ]]; then
    cp "$INDEX_JS" bin/assets/index.js
    print_success "Copied $(basename "$INDEX_JS") as index.js"
fi

if [[ -n "$INDEX_CSS" ]]; then
    cp "$INDEX_CSS" bin/assets/index.css
    print_success "Copied $(basename "$INDEX_CSS") as index.css"
fi

# Copy MuJoCo WASM assets with generic names
MUJOCO_WASM=$(ls dist/assets/mujoco_wasm-*.wasm 2>/dev/null | head -1)
MUJOCO_JS=$(ls dist/assets/mujoco_wasm-*.js 2>/dev/null | head -1)

if [[ -n "$MUJOCO_WASM" ]]; then
    cp "$MUJOCO_WASM" bin/assets/mujoco_wasm.wasm
    print_success "Copied $(basename "$MUJOCO_WASM") as mujoco_wasm.wasm"
fi

if [[ -n "$MUJOCO_JS" ]]; then
    cp "$MUJOCO_JS" bin/assets/mujoco_wasm.js
    print_success "Copied $(basename "$MUJOCO_JS") as mujoco_wasm.js"
fi

print_success "Essential app files copied to bin directory with generic names"

print_success "🎉 Light build completed successfully!"
echo ""
echo -e "${GREEN}📁 Built files are in: ./dist${NC}"
echo -e "${GREEN}🚀 To start the server: npm start${NC}"
echo -e "${GREEN}🌐 Server will run on: http://localhost:7860${NC}"
echo ""
echo -e "${YELLOW}💡 Note: This was a light build (React only)${NC}"
echo -e "${YELLOW}💡 If you need to rebuild MuJoCo WASM, use: ./build.sh${NC}"