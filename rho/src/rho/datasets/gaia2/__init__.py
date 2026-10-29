from __future__ import annotations

__all__ = ["Gaia2Dataset"]


def __getattr__(name: str):
    if name == "Gaia2Dataset":
        from rho.datasets.gaia2.dataset import Gaia2Dataset

        return Gaia2Dataset
    raise AttributeError(name)
