import ast
from typing import List

'''
Utility file needed to parse and create the syntax tree from textual data
'''

def create_syntax_tree(fn_name: str, out_name: str, in_names: List[str], keywords: List = None) -> ast.AST:
    fn_args: str = ""
    for name in in_names:
        fn_args += "%s," % name
    fn_statement = "%s = %s(%s)" % (out_name, fn_name, fn_args)
    fn_syn_tree = ast.parse(fn_statement, "exec").body
    if keywords is not None:
        fn_syn_tree[0].value.keywords = keywords
    return fn_syn_tree


def define_syntax_tree(fn_name: str, out_name: str, in_names: List[str], model_param: str) -> ast.AST:
    argms: str = ""
    for name in in_names:
        argms += "%s," % name
    statement = "%s = %s(%s, [%s])" % (out_name, fn_name, model_param, argms)
    return ast.parse(statement, "exec").body