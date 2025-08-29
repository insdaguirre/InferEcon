import importlib
import pkgutil
from typing import Callable, Dict, List, Protocol


class AnalysisFunction(Protocol):
    display_name: str

    def apply(self, df):  # pandas.DataFrame, but avoid hard import here
        ...


def load_functions() -> List[AnalysisFunction]:
    """Dynamically discover analysis functions in this package.

    Any module in this package whose name ends with `_func` and exposes
    `display_name: str` and `apply(df)` will be loaded.
    """
    functions: List[AnalysisFunction] = []
    package_name = __name__
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__, package_name + "."):
        module_name = module_info.name
        if not module_name.endswith("_func"):
            continue
        module = importlib.import_module(module_name)
        if hasattr(module, "display_name") and hasattr(module, "apply"):
            functions.append(module)  # type: ignore[arg-type]
    # Sort by name for stable UI
    functions.sort(key=lambda m: getattr(m, "display_name", ""))
    return functions
