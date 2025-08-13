"""DS Template Package.

A structured package for data science workflows.
"""

__version__ = "0.1.0"
__author__ = "Data Science Team"
__email__ = "team@company.com"

# Package-level imports
from . import features
from . import models

__all__ = ['features', 'models', '__version__']
