import inspect

def get_full_obj_type(obj):
    obj_mro = inspect.getmro(obj.__class__)
    assert len(obj_mro) >= 2
    base = obj_mro[-2]
    name = f"{base.__module__}.{base.__name__}"
    return name