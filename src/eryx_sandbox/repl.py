"""Running code REPL style in an eryx session, without any agent framework."""

import ast
import re

_ECHO_VAR = "__eryx_repl_value"


def prepare_cell(code: str) -> str:
    """Return `code` with a bare last expression rewritten to print its repr.

    Source lines are spliced rather than unparsed, so line numbers in tracebacks
    still match the code as written. Code that doesn't parse is returned unchanged
    so the sandbox reports the SyntaxError.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    # only the line breaks ast counts: str.splitlines() also splits on \x0c, \u2028...
    lines = re.split(r"\r\n|\r|\n", code)
    start, end = last.lineno - 1, last.end_lineno or last.lineno
    # what precedes the expression on its line, e.g. `x = 1; ` (offsets are in bytes)
    head = lines[start].encode()[: last.col_offset].decode()
    expr = ast.get_source_segment(code, last)
    echo = (
        f"{head}{_ECHO_VAR} = ({expr}\n)\n"
        f"if {_ECHO_VAR} is not None:\n    print(repr({_ECHO_VAR}))\n"
        f"del {_ECHO_VAR}"
    )
    return "\n".join([*lines[:start], echo, *lines[end:]])
