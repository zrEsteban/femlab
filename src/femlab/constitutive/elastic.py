from __future__ import annotations
import numpy as np

def D_plane_stress(E: float, nu: float) -> np.ndarray:
    c = E / (1.0 - nu**2)
    return c * np.array([[1, nu, 0],
                         [nu, 1, 0],
                         [0,  0, (1-nu)/2]])
