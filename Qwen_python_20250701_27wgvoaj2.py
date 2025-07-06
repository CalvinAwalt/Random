import inspect
import ast
from functools import wraps

def self_modify(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        src = inspect.getsource(fn)
        tree = ast.parse(src)
        # Modify AST here (simplified example)
        new_code = compile(tree, filename="<ast>", mode="exec")
        exec(new_code)
        return fn(*args, **kwargs)
    return wrapper

@self_modify
def adaptive_function(x):
    return x * 2