# Temporary: adds src/ to sys.path so tests can import lichtheim2 without
# an editable install. Remove once pyproject.toml / pip install -e . is added.
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
