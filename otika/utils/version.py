from .. import __version__
from ..protocol import Version


def get_version():
    version_split = __version__.split(".")
    return Version(
        major=int(version_split[0]),
        minor=int(version_split[1]),
        patch=int(version_split[2]),
    )


def compare_version(version: Version, other: Version) -> int:
    if version.major > other.major:
        return 1
    elif version.major < other.major:
        return -1
    else:
        if version.minor > other.minor:
            return 1
        elif version.minor < other.minor:
            return -1
        else:
            if version.patch > other.patch:
                return 1
            elif version.patch < other.patch:
                return -1
            else:
                return 0
