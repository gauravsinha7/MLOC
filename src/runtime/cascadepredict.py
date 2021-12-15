import ast
import copy
import inspect
from typing import Mapping, Callable

import numpy as np
import pandas as pd
import scipy

from src.graph.utility import create_syntax_tree, define_syntax_tree
from src.graph.node import GraphNode


def predict_cascades(func: Callable, model_node: GraphNode,
                     predict_function: Callable, predict_proba_function: Callable,
                     cascades_dict: Mapping) -> Callable:
    class CascadePredict(ast.NodeTransformer):
        def visit_FunctionDef(self, orig_ast: ast.FunctionDef) -> ast.AST:
            cascades_body = []
            # Compute selected features.
            for node in selected_feature_nodes:
                cascades_body += node.get_syntax_tree()
            # Predict with approximate model.
            approximate_model_ast = define_syntax_tree(fn_name=predict_proba_function.__name__,
                                                     out_name="__saomls_approximate_preds",
                                                     in_names=selected_feature_names,
                                                     model_param="__saomls_approximate_model")
            cascades_body += approximate_model_ast
            # Get indices that can't be approximated.
            unapproximated_indices_ast = create_syntax_tree(fn_name="__saomls_get_unapproximated_indices",
                                                             out_name="__saomls_unapproximated_indices",
                                                             in_names=["__saomls_approximate_preds",
                                                                          "__saomls_cascade_threshold"])
            cascades_body += unapproximated_indices_ast
            # Only compute remaining features for unapproximated indices.
            shortened_inputs = set()
            for node in remaining_feature_nodes:
                for input_name in node.in_names:
                    if input_name not in shortened_inputs:
                        shorten_ast = create_syntax_tree(fn_name="__saomls_select_unapproximated_rows",
                                                          out_name=input_name,
                                                          in_names=[input_name, "__saomls_unapproximated_indices"])
                        cascades_body += shorten_ast
                        shortened_inputs.add(input_name)
                cascades_body += node.get_syntax_tree()
            # Shorten the selected features.
            for name in selected_feature_names:
                shorten_ast = create_syntax_tree(fn_name="__saomls_select_unapproximated_rows",
                                                  out_name=name,
                                                  in_names=[name, "__saomls_unapproximated_indices"])
                cascades_body += shorten_ast
            # Predict with full model.
            full_model_ast = define_syntax_tree(fn_name=predict_function.__name__,
                                              out_name="__saomls_full_preds",
                                              in_names=model_node.in_names,
                                              model_param="__saomls_full_model")
            cascades_body += full_model_ast
            # Return combined predictions.
            combine_predictions_ast = create_syntax_tree(fn_name="__saomls_combine_predictions",
                                                          out_name="__saomls_final_predictions",
                                                          in_names=["__saomls_approximate_preds",
                                                                       "__saomls_full_preds",
                                                                       "__saomls_cascade_threshold"])
            cascades_body += combine_predictions_ast
            return_ast = ast.parse("return __saomls_final_predictions", "exec").body
            cascades_body += return_ast
            # Finalize AST.
            new_ast = copy.deepcopy(orig_ast)
            new_ast.body = cascades_body
            # No recursion allowed!
            new_ast.decorator_list = []
            return ast.copy_location(new_ast, orig_ast)

    func_source = inspect.getsource(func)
    func_ast = ast.parse(func_source)
    selected_feature_indices = cascades_dict["selected_feature_indices"]
    selected_feature_nodes = [model_node.input_nodes[i] for i in selected_feature_indices]
    selected_feature_names = [model_node.in_names[i] for i in selected_feature_indices]
    remaining_feature_nodes = [model_node.input_nodes[i] for i in range(len(model_node.input_nodes)) if
                               i not in selected_feature_indices]
    cascades_transformer = CascadePredict()
    cascades_ast = cascades_transformer.visit(func_ast)
    cascades_ast = ast.fix_missing_locations(cascades_ast)
    # import astor
    # print(astor.to_source(cascades_ast))
    # Create namespaces the instrumented function can run in containing both its
    # original globals and the ones the instrumentation needs.
    local_namespace = {}
    augmented_globals = copy.copy(func.__globals__)
    augmented_globals["__saomls_approximate_model"] = cascades_dict["approximate_model"]
    augmented_globals["__saomls_full_model"] = cascades_dict["full_model"]
    augmented_globals["__saomls_cascade_threshold"] = cascades_dict["cascade_threshold"]
    augmented_globals["__saomls_get_unapproximated_indices"] = get_unapproximated_indices
    augmented_globals["__saomls_select_unapproximated_rows"] = select_unapproximated_rows
    augmented_globals["__saomls_combine_predictions"] = combine_predictions
    # Run the instrumented function.
    exec(compile(cascades_ast, filename="<ast>", mode="exec"), augmented_globals,
         local_namespace)
    return local_namespace[func.__name__]


def get_unapproximated_indices(approximated_preds, cascade_threshold):
    return np.logical_and(approximated_preds < cascade_threshold,
                          approximated_preds > 1 - cascade_threshold).nonzero()[0]


def select_unapproximated_rows(input_data, unapproximated_indices):
    if isinstance(input_data, scipy.sparse.csr.csr_matrix) or isinstance(input_data, np.ndarray):
        return input_data[unapproximated_indices]
    elif isinstance(input_data, pd.DataFrame):
        return input_data.iloc[unapproximated_indices].reset_index().drop("index", axis=1)
    else:
        return input_data


def combine_predictions(approximate_predictions, full_predictions, cascade_threshold):
    final_predictions = np.zeros(approximate_predictions.shape, dtype=full_predictions.dtype)
    full_prediction_index = 0
    for i in range(len(final_predictions)):
        if approximate_predictions[i] >= cascade_threshold:
            final_predictions[i] = 1
        elif approximate_predictions[i] <= 1 - cascade_threshold:
            final_predictions[i] = 0
        else:
            final_predictions[i] = full_predictions[full_prediction_index]
            full_prediction_index += 1
    assert(full_prediction_index == len(full_predictions))
    return final_predictions
