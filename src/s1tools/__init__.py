from importlib.metadata import version

try:
    __version__ = version("s1tools")
except Exception:
    __version__ = "999"
