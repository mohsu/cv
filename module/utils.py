# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: utils.py 
@time: 2020/06/16
"""

# python packages
from functools import reduce

# 3rd-party packages
from loguru import logger


# self-defined packages
@logger.catch(reraise=True)
def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
