"""Bounded pure-function hidden-test verifier; no model-supplied imports or I/O."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import resource
import sys


BUILTINS = {name: value for name, value in {
    'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip, 'min': min, 'max': max,
    'sum': sum, 'sorted': sorted, 'reversed': reversed, 'list': list, 'tuple': tuple,
    'set': set, 'dict': dict, 'int': int, 'str': str, 'bool': bool, 'abs': abs,
    'all': all, 'any': any, 'isinstance': isinstance,
}.items()}
METHODS = {'append', 'extend', 'add', 'discard', 'intersection', 'union', 'difference', 'issubset', 'issuperset',
           'pop', 'sort', 'reverse', 'count', 'index', 'lower', 'upper', 'strip', 'split', 'join',
           'fromkeys', 'get', 'keys', 'values', 'items', 'copy', 'update', 'remove', 'clear', 'insert'}
FORBIDDEN = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.With, ast.AsyncWith, ast.AsyncFunctionDef,
             ast.Await, ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal, ast.Lambda)


def check(source, tests):
    if not isinstance(source, str) or len(source) > 16000 or not isinstance(tests, list) or len(tests) > 64:
        raise ValueError('Bounded source and tests required')
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    if len(nodes) > 2000 or any(isinstance(node, FORBIDDEN) for node in nodes):
        raise ValueError('Unsupported code syntax')
    for node in nodes:
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('Private runtime names are unavailable')
        if isinstance(node, ast.Attribute) and node.attr not in METHODS:
            raise ValueError('Only bounded collection/string methods are available')
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name != 'solve'):
            raise ValueError('Define only solve without decorators')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in {*BUILTINS, 'solve'}:
            raise ValueError('Unsupported callable')
    namespace = {'__builtins__': BUILTINS}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(tree, '<candidate-solution>', 'exec'), namespace)
        function = namespace.get('solve')
        if not callable(function):
            raise ValueError('Missing solve function')
        outcomes = []
        for test in tests:
            try:
                # Copy inputs so mutation cannot change a later hidden oracle.
                args = json.loads(json.dumps(test['args']))
                outcomes.append(function(*args) == test['expected'])
            except Exception:
                outcomes.append(False)
    return {'passed': all(outcomes), 'passed_tests': sum(outcomes), 'total_tests': len(outcomes)}


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    try:
        value = json.loads(sys.stdin.read(65537))
        result = check(value['source'], value['tests'])
    except Exception as exc:
        result = {'passed': False, 'reason': type(exc).__name__}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
