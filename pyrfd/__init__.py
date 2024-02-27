""" Random function descent, covariance models and their estimation
"""

from .optimizer import RFD
from .covariance import SquaredExponential

__all__ = ["RFD", "SquaredExponential"]
