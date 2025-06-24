__all__ = ["__version__", "__version_tuple__", "version", "version_tuple"]

TYPE_CHECKING = False
VERSION_TUPLE = tuple[int | str, ...] if TYPE_CHECKING else object

version: str
__version__: str
__version_tuple__: VERSION_TUPLE
version_tuple: VERSION_TUPLE

__version__ = version = "0.1.dev12+g995430a.d20250522"
__version_tuple__ = version_tuple = (0, 1, "dev12", "g995430a.d20250522")
