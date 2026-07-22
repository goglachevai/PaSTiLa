import platform
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

_IS_WINDOWS = platform.system() == "Windows"
_PDSS_NAME = "PDSS.exe" if _IS_WINDOWS else "PDSS"
_PSF_NAME = "PSF.exe" if _IS_WINDOWS else "PSF"

DEFAULT_PDSS_EXE = PROJECT_DIR / "algorithms" / "PDSS" / _PDSS_NAME
DEFAULT_PSF_EXE = PROJECT_DIR / "algorithms" / "PSF" / _PSF_NAME
DEFAULT_MODEL_DIR = PROJECT_DIR / "models"
