"""Math tools — calculator and sandboxed Python executor.

Both tools are designed to be safe:
- `calculator` uses regex whitelist for characters
- `python_executor` uses restricted namespace + forbidden-pattern check
"""
from __future__ import annotations

import math
import re

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, (), sqrt(), pow(), abs().
    Examples: '2 + 3 * 4', '(15 * 37) + 42', '2**10', 'sqrt(144)'
    Use this for ANY math question — never compute in your head.
    """
    try:
        safe_expr = expression.strip()
        # Whitelist: digits, operators, parens, dots, spaces, math functions
        allowed_pattern = re.compile(r"^[0-9+\-*/.() ,sqrtpowabsminmax]+$")
        if not allowed_pattern.match(safe_expr):
            return (
                "Error: Expression contains invalid characters. "
                "Use only numbers and operators (+, -, *, /, **, ())"
            )

        # Map user-friendly names to math module
        safe_expr = safe_expr.replace("sqrt", "math.sqrt")
        safe_expr = safe_expr.replace("pow", "math.pow")
        safe_expr = safe_expr.replace("abs", "abs")

        result = eval(
            safe_expr,
            {"__builtins__": {}, "math": math, "abs": abs},
            {},
        )
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def python_executor(code: str) -> str:
    """Execute simple Python code for data processing or calculations.
    Only basic operations allowed (no imports, no file I/O, no network).
    Use for complex calculations, string processing, list/dict manipulation,
    or any logic that needs more than the calculator tool can express.

    The sandbox allows: len, str, int, float, list, dict, range, enumerate,
    sorted, sum, min, max, abs, round, zip, map, filter, print, type, isinstance.
    """
    try:
        forbidden = [
            "import", "open(", "exec(", "eval(", "__", "os.", "sys.",
            "subprocess", "requests", "http", "socket",
        ]
        for f in forbidden:
            if f in code:
                return f"Error: '{f}' is not allowed for security reasons."

        namespace: dict = {
            "__builtins__": {
                "len": len, "str": str, "int": int, "float": float,
                "list": list, "dict": dict, "range": range, "enumerate": enumerate,
                "sorted": sorted, "sum": sum, "min": min, "max": max, "abs": abs,
                "round": round, "zip": zip, "map": map, "filter": filter,
                "print": print, "type": type, "isinstance": isinstance,
            }
        }
        exec(code, namespace)

        if "result" in namespace:
            return f"Result: {namespace['result']}"
        return "Code executed successfully (no 'result' variable set)."
    except Exception as e:
        return f"Error: {e}"


__all__ = ["calculator", "python_executor"]
