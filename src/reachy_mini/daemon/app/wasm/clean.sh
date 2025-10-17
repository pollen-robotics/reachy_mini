#!/bin/bash

# Clean script for MuJoCo WASM React App
# Removes ALL build artifacts and returns to clean state

set -e  # Exit on any error

echo "🧹 Starting complete cleanup..."

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

print_warning "Removing ALL build artifacts and dependencies..."
echo "Files being removed:"
echo "  - dist/"
echo "  - node_modules/"
echo "  - package.json"
echo "  - package-lock.json"
echo "  - pnpm-lock.yaml"
echo "  - build_workspace/"
echo "  - emsdk/"
echo "  - Any CMake build directories"
echo ""

print_step "Removing distribution files..."
if [[ -d "dist" ]]; then
    rm -rf dist
    print_success "Removed dist/"
else
    print_warning "dist/ not found"
fi

print_step "Removing Node.js dependencies..."
if [[ -d "node_modules" ]]; then
    rm -rf node_modules
    print_success "Removed node_modules/"
else
    print_warning "node_modules/ not found"
fi

print_step "Removing package files..."
if [[ -f "package.json" ]]; then
    rm -f package.json
    print_success "Removed package.json"
else
    print_warning "package.json not found"
fi

if [[ -f "package-lock.json" ]]; then
    rm -f package-lock.json
    print_success "Removed package-lock.json"
else
    print_warning "package-lock.json not found"
fi

if [[ -f "pnpm-lock.yaml" ]]; then
    rm -f pnpm-lock.yaml
    print_success "Removed pnpm-lock.yaml"
else
    print_warning "pnpm-lock.yaml not found"
fi

print_step "Removing build workspace..."
if [[ -d "build_workspace" ]]; then
    rm -rf build_workspace
    print_success "Removed build_workspace/"
else
    print_warning "build_workspace/ not found"
fi

print_step "Removing Emscripten SDK..."
if [[ -d "emsdk" ]]; then
    rm -rf emsdk
    print_success "Removed emsdk/"
else
    print_warning "emsdk/ not found"
fi

print_step "Cleaning MuJoCo web fork build artifacts..."
if [[ -d "fork/mujoco_web/build" ]]; then
    rm -rf fork/mujoco_web/build
    print_success "Removed fork/mujoco_web/build/"
else
    print_warning "fork/mujoco_web/build/ not found"
fi

if [[ -d "fork/mujoco_web/dist" ]]; then
    rm -rf fork/mujoco_web/dist
    print_success "Removed fork/mujoco_web/dist/"
else
    print_warning "fork/mujoco_web/dist/ not found"
fi

if [[ -d "fork/mujoco_web/node_modules" ]]; then
    rm -rf fork/mujoco_web/node_modules
    print_success "Removed fork/mujoco_web/node_modules/"
else
    print_warning "fork/mujoco_web/node_modules/ not found"
fi

if [[ -d "fork/mujoco_web/wasm_backup" ]]; then
    rm -rf fork/mujoco_web/wasm_backup
    print_success "Removed fork/mujoco_web/wasm_backup/"
else
    print_warning "fork/mujoco_web/wasm_backup/ not found"
fi

if [[ -d "fork/mujoco_web/public" ]]; then
    rm -rf fork/mujoco_web/public
    print_success "Removed fork/mujoco_web/public/"
else
    print_warning "fork/mujoco_web/public/ not found"
fi

if [[ -f "fork/mujoco_web/pnpm-lock.yaml" ]]; then
    rm -f fork/mujoco_web/pnpm-lock.yaml
    print_success "Removed fork/mujoco_web/pnpm-lock.yaml"
else
    print_warning "fork/mujoco_web/pnpm-lock.yaml not found"
fi

print_step "Cleaning WASM build artifacts..."
if [[ -d "fork/mujoco_web/src/wasm" ]]; then
    find fork/mujoco_web/src/wasm -name "*.wasm" -delete 2>/dev/null || true
    find fork/mujoco_web/src/wasm -name "*.js" -delete 2>/dev/null || true
    find fork/mujoco_web/src/wasm -name "*.wasm.map" -delete 2>/dev/null || true
    print_success "Cleaned WASM artifacts from fork/mujoco_web/src/wasm/"
fi

print_step "Removing any temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.temp" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true

print_step "Cleaning CMake caches..."
find . -name "CMakeCache.txt" -delete 2>/dev/null || true
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "cmake_install.cmake" -delete 2>/dev/null || true
find . -name "Makefile" -not -path "./fork/mujoco_web/Makefile" -delete 2>/dev/null || true

print_success "Temporary files cleaned"

# Reset any environment variables that might have been set
print_step "Resetting environment variables..."
unset EMSDK 2>/dev/null || true
print_success "Environment variables reset"

print_success "🎉 Complete cleanup finished!"
echo ""
echo -e "${GREEN}🧼 Repository is now in clean state${NC}"
echo -e "${GREEN}📁 Only source files remain (Dockerfile, server.js, fork/, stewart_little_control/, etc.)${NC}"
echo -e "${GREEN}🚀 Ready for fresh build with: ./build.sh${NC}"