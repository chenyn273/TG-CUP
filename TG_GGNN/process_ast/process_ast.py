import re
from process_ast.ast_diff_parser import get_diff_ast


def get_ast_diff_json(old_method, new_method):
    with open('./process_ast/old.java', 'w') as _f:
        _f.write(old_method)
        _f.close()
    with open('./process_ast/new.java', 'w') as _f:
        _f.write(new_method)
        _f.close()
    ast_diff = get_diff_ast("./process_ast/old.java",
                            "./process_ast/new.java",
                            "./process_ast/action.json",
                            "./process_ast/ast-diffing-1.6-jar-with-dependencies.jar")
    return ast_diff.to_json()


