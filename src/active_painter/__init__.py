"""Strict active-inference painting scaffold."""

from .config import PainterConfig
from .env import PaintCanvasEnv, StrokeAction
from .agent import ActiveInferencePainter
from .version import package_version

__version__ = package_version()

__all__ = ["PainterConfig", "PaintCanvasEnv", "StrokeAction", "ActiveInferencePainter", "__version__"]
