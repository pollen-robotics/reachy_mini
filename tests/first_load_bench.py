import time
import toml
import importlib
from pathlib import Path

module_to_name = {
    "opencv-python": "cv2",
    "eclipse-zenoh": "zenoh",
    "pyserial": "serial",
    "pyusb": "usb",
}

# Locate parent pyproject.toml
pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

# Read dependencies from pyproject.toml
with open(pyproject_path, "r") as f:
    pyproject = toml.load(f)

# Try to get dependencies from [project]
deps = []
if "project" in pyproject and "dependencies" in pyproject["project"]:
    deps = pyproject["project"]["dependencies"]

# Clean dependency names (remove version specifiers)
def clean_dep(dep):
    if isinstance(dep, str):
        dep = dep.split()[0].split('=')[0].split('<')[0].split('>')[0]

        # remove [stuff] extras
        if "[" in dep:
            dep = dep.split("[")[0]

        if dep in module_to_name:
            return module_to_name[dep]
        
        dep = dep.replace("-", "_")

    return dep

deps = [clean_dep(dep) for dep in deps if dep.lower() != "python"]

print("Timing imports for dependencies:")

total = 0.0
for dep in deps:
    try:
        start = time.perf_counter()
        importlib.import_module(dep)
        elapsed = time.perf_counter() - start
        print(f"{dep:30s}: {elapsed:.4f} s")
        total += elapsed
    except Exception as e:
        print(f"{dep:30s}: FAILED ({e})")

print(f"{'Total':30s}: {total:.4f} s")