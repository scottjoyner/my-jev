import importlib
import pkgutil

import my_jev


def test_every_module_imports():
    """A module nothing imports can still be silently broken.

    Removing a helper from `train` left `train_causal` unimportable, and the
    suite stayed green because no test imported it. Walk the package instead.
    """
    failures = {}
    for module in pkgutil.iter_modules(my_jev.__path__):
        name = f"my_jev.{module.name}"
        try:
            importlib.import_module(name)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"

    assert not failures, f"modules failed to import: {failures}"