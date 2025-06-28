import inspect
from functools import wraps


def register_init_params(cls):
    """
    Decorator to add an `init_params` dictionary to any class,
    storing all `__init__` arguments by their parameter names.
    """
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # Get the parameter names of the original __init__ method
        sig = inspect.signature(original_init)
        param_names = list(sig.parameters.keys())[1:]  # Exclude 'self'

        # Map args and kwargs to their corresponding parameter names
        init_params = {}
        for name, value in zip(param_names, args):
            init_params[name] = value
        init_params.update(kwargs)

        # Call the original __init__ method
        original_init(self, *args, **kwargs)

        # Store the mapped parameters in the instance
        setattr(self, '_init_params', init_params)

    cls.__init__ = new_init
    return cls