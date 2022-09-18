from codelets.adl.graph import ArchitectureNode
import polymath as pm
import inspect
CLASS_ATTR = {}
CLASS_ATTR[f'NodePlaceholder'] = dir(pm.Node)
CLASS_ATTR[f'HAGPlaceholder'] = dir(ArchitectureNode)

TEMPLATE_CLASS_MAP = {}
TEMPLATE_CLASS_MAP['NodePlaceholder'] = f"{inspect.getmro(pm.Node)[-2].__module__}.{inspect.getmro(pm.Node)[-2].__name__}"
TEMPLATE_CLASS_MAP['HAGPlaceholder'] = f"{inspect.getmro(ArchitectureNode)[-2].__module__}.{inspect.getmro(ArchitectureNode)[-2].__name__}"

CLASS_TEMPLATE_MAP = {v: k for k, v in TEMPLATE_CLASS_MAP.items()}

TEMPLATE_CLASS_ARG_MAP = {}
TEMPLATE_CLASS_ARG_MAP['NodePlaceholder'] = ['node']
TEMPLATE_CLASS_ARG_MAP['HAGPlaceholder'] = ['hag']