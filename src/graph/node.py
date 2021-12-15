#import for abstract syntax tree
import ast  

from typing import List
import src.graph.utility

'''
This file depicts the nodes in the graph for performing the cascading operations
agnostically.
'''

class GraphNode:

    def __init__(self, fn_name: str, out_name: str, in_names: List[str],
                 input_nodes: List['GraphNode'], cost: float, keywords: List = None, param: str = None):
        self.fn_name = fn_name
        self.out_name = out_name
        self.in_names = in_names
        self.input_nodes = input_nodes
        self.cost = cost
        assert(len(input_nodes) == len(in_names))
        self.param = param
        self.keywords = keywords

    def get_syntax_tree(self) -> ast.AST:
        if self.param is None:
            return src.graph.utility.create_syntax_tree(fn_name=self.fn_name,
                                                               out_name=self.out_name,
                                                               in_names=self.in_names,
                                                               keywords=self.keywords)
        else:
            return src.graph.utility.define_syntax_tree(fn_name=self.fn_name,
                                                            out_name=self.out_name,
                                                            in_names=self.in_names,
                                                            param=self.param)