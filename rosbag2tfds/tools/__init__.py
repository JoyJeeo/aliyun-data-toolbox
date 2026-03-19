"""Helper modules for converting ROS bags to Open-X TFDS datasets."""

from importlib import metadata


def version() -> str:
    """Return package version if installed, otherwise use '0.0.0'."""
    try:
        return metadata.version("openx_delivery_tools")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__all__ = ["version"]

