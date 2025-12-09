"""Check system requirements for package management."""
import logging
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def check_requirement(name, cmd):
    """Check if a requirement is met. Returns (bool, str)."""
    if not shutil.which(name):
        return False, "not found"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            # Return the full output line, stripped
            version = result.stdout.strip().split('\n')[0] if result.stdout else "unknown"
            return True, version
        return False, "error running"
    except Exception as e:
        return False, f"error: {e}"


def main():
    """Check and report on system requirements for package management."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    checks = {
        'uv': (['uv', '--version'], True),
        'git': (['git', '--version'], True),
        'git-lfs': (['git', 'lfs', 'version'], False),
    }
    
    results = {}
    for name, (cmd, required) in checks.items():
        ok, info = check_requirement(name, cmd)
        results[name] = ok
        
        if ok:
            logger.info(f"{name}: {info}")
        elif required:
            logger.error(f"{name}: {info} (required)")
        else:
            logger.warning(f"{name}: {info} (optional)")
    
    if not all(results[k] for k in ['uv', 'git']):
        logger.error("Missing required dependencies")
        if not results['uv']:
            logger.error("Install uv: pip install uv")
        if not results['git']:
            logger.error("Install git: https://git-scm.com/downloads")
        return 1
    
    if not results['git-lfs']:
        logger.info("git-lfs not found but optional - install for better large file support")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
