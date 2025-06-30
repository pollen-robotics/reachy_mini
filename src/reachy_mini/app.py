from abc import ABC, abstractmethod
from pathlib import Path
import threading

from jinja2 import Environment, FileSystemLoader

from reachy_mini.reachy_mini import ReachyMini


class ReachyMiniApp(ABC):
    def __init__(self):
        self.stop_event = threading.Event()

    def wrapped_run(self):
        try:
            with ReachyMini() as reachy_mini:
                self.run(reachy_mini, self.stop_event)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    @abstractmethod
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the main logic of the app."""
        pass

    def stop(self):
        """Stop the app gracefully."""
        self.stop_event.set()
        print("App is stopping...")


def make_app_project(app_name: str, path: Path):
    TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    def render_template(filename, context):
        template = env.get_template(filename)
        return template.render(context)

    base_path = Path(path).resolve() / app_name
    if base_path.exists():
        print(f"❌ Folder {base_path} already exists.")
        return

    module_name = app_name.replace("-", "_")
    class_name = "".join(word.capitalize() for word in module_name.split("_"))

    base_path.mkdir()
    (base_path / module_name).mkdir()

    # Generate files
    context = {
        "app_name": app_name,
        "package_name": app_name,
        "module_name": module_name,
        "class_name": class_name,
    }

    (base_path / module_name / "__init__.py").touch()
    (base_path / module_name / "main.py").write_text(
        render_template("main.py.j2", context)
    )
    (base_path / "pyproject.toml").write_text(
        render_template("pyproject.toml.j2", context)
    )
    (base_path / "README.md").write_text(render_template("README.md.j2", context))

    print(f"✅ Created app in {base_path}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a new Reachy Mini app project."
    )
    parser.add_argument("app_name", type=str, help="Name of the app to create.")
    parser.add_argument(
        "path",
        type=Path,
        help="Path where the app project will be created.",
    )

    args = parser.parse_args()
    make_app_project(args.app_name, args.path)


if __name__ == "__main__":
    main()
