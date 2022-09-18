FXP_CONFIGS = {
    "FXP32": {"signed": True, "n_int": 15, "n_frac": 16, "overflow": "saturate", "n_word": 32}
}
from .common.datatype import Datatype
from . import graph
from . import adl
from .adl import util
from .compiler import initialize_program
