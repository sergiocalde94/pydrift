from functools import wraps


def check_optional_module(_func=None,
                          *,
                          has_module: bool,
                          exception_message: str):
    def decorator_check_optional_module(func):
        @wraps(func)
        def wrapper_check_optional_module(*args, **kwargs):
            if not has_module:
                raise ModuleNotFoundError(exception_message)
            return func(*args, **kwargs)
        return wrapper_check_optional_module

    if _func is None:
        return decorator_check_optional_module
    else:
        return decorator_check_optional_module(_func)
