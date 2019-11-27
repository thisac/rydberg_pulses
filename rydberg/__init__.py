from .utils import print_array, save_dict

from .system.system import System, get_basis

__all__ = [
    "__title__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "print_array",
    "save_dict",
    "System",
    "get_basis",
    ]

__title__ = "Rydberg pulse optimizer"
__version__ = "0.4"
__description__ = "Pulse sequence optimizer for neutral atom arrays"
__url__ = "https://github.com/thisac/rydberg"

__author__ = "Theodor Isacsson"
__email__ = "isacsson@mit.edu"

__license__ = "MIT"
