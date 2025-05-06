import ast
import re
from pathlib import Path
import textwrap

PROCESSING_PATH = Path("nmr_fido/core/processing.py")
NMRDATA_PATH = Path("nmr_fido/nmrdata.py")
INIT_PATH = Path("nmr_fido/__init__.py")


def inject_alias_docstrings() -> None:
    """
    Extract aliases like ZF = zero_fill and
    add __doc__/__name__ assignments.
    """
    source = PROCESSING_PATH.read_text()
    lines = source.splitlines()

    tree = ast.parse(source)
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    alias_pattern = re.compile(r"^([A-Z]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)$")
    
    updated_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        updated_lines.append(line)

        match = alias_pattern.match(line.strip())
        if match:
            alias, target = match.groups()

            # Only process if target is a known function
            if target in function_names:
                next_lines = lines[i + 1:i + 3]
                needs_doc = not any(f"{alias}.__doc__" in l for l in next_lines)
                needs_name = not any(f"{alias}.__name__" in l for l in next_lines)

                if needs_doc:
                    updated_lines.append(f"{alias}.__doc__ = {target}.__doc__  # Auto-generated")
                if needs_name:
                    updated_lines.append(f'{alias}.__name__ = "{alias}"  # Auto-generated')

        i += 1

    # Only write if changes occurred
    if updated_lines != lines:
        PROCESSING_PATH.write_text("\n".join(updated_lines))


def expose_processing_in_init() -> None:
    """Regenerate __init__.py imports and __all__ for all public processing functions and aliases."""

    # Load function + alias names from processing.py
    processing_source = PROCESSING_PATH.read_text()
    tree = ast.parse(processing_source)

    functions = []
    alias_map = {}  # key = function, value = [alias1, alias2, ...]

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            functions.append(node.name)
        elif isinstance(node, ast.Assign):
            if (
                isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Name)
                and node.targets[0].id.isupper()
                and node.value.id in functions
            ):
                alias = node.targets[0].id
                func = node.value.id
                alias_map.setdefault(func, []).append(alias)

    # Build the import section
    import_block = [
        "from .nmrdata import NMRData",
        "from .core.processing import ("
    ]
    for func in functions:
        aliases = alias_map.get(func, [])
        if aliases:
            import_block.append(f"    {func}, " + ", ".join(aliases) + ",")
        else:
            import_block.append(f"    {func},")
    import_block.append(")")
    import_block.append("")

    # Build the __all__ block
    all_block = ["__all__ = ["]
    all_block.append('    "NMRData",')
    for func in functions:
        parts = [f'"{func}"'] + [f'"{alias}"' for alias in alias_map.get(func, [])]
        all_block.append("    " + ", ".join(parts) + ",")
    all_block.append("]")

    # Replace the relevant section in __init__.py
    init_lines = INIT_PATH.read_text().splitlines()

    anchor = init_lines.index("from .nmrdata import NMRData")

    new_init = init_lines[:anchor] + import_block + all_block
    INIT_PATH.write_text("\n".join(new_init))


def inject_stubs_into_NMRData(include_docstrings: bool = True) -> None:
    """
    Inject method stubs for processing functions and aliases into the NMRData
    class for IDE visibility.
    """
    
    processing_source = PROCESSING_PATH.read_text()
    processing_ast = ast.parse(processing_source)

    func_defs = {}
    aliases = {}

    # Parse functions and their full signatures (with defaults and type hints)
    for node in processing_ast.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            func_name = node.name
            args = []
            total_args = node.args.args
            total_defaults = [None] * (len(total_args) - len(node.args.defaults)) + node.args.defaults

            # Positional args (skip first 'data')
            for arg, default in zip(total_args[1:], total_defaults[1:]):
                s = arg.arg
                if arg.annotation:
                    s += f": {ast.unparse(arg.annotation)}"
                if default:
                    s += f" = {ast.unparse(default)}"
                args.append(s)

            # Keyword-only args
            if node.args.kwonlyargs:
                if not args or args[-1] != "*":
                    args.append("*")
                for kwarg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                    s = kwarg.arg
                    if kwarg.annotation:
                        s += f": {ast.unparse(kwarg.annotation)}"
                    if default:
                        s += f" = {ast.unparse(default)}"
                    args.append(s)


            arg_string = ", ".join(args)
            docstring = ast.get_docstring(node)

            if include_docstrings and docstring:
                doc_lines = ['    """'] + [f"    {line}" for line in docstring.splitlines()] + ['    """']
                stub = [f"def {func_name}(self, {arg_string}) -> NMRData:"] + doc_lines + ["    ..."]
            else:
                stub = [f"def {func_name}(self, {arg_string}) -> NMRData:", "    ..."]

            func_defs[func_name] = stub

    # Parse aliases
    for line in processing_source.splitlines():
        match = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)$", line.strip())
        if match:
            alias, target = match.groups()
            if target in func_defs:
                aliases.setdefault(target, []).append(alias)

    # Combine stubs (grouped)
    grouped_stubs = []
    for func_name, stub_lines in func_defs.items():
        grouped_stubs.extend(stub_lines)
        for alias in aliases.get(func_name, []):
            alias_stub = [line.replace(f"def {func_name}(", f"def {alias}(") for line in stub_lines]
            grouped_stubs.extend(alias_stub)
        grouped_stubs.append("")  # spacing between groups

    # Insert into NMRData
    nmrdata_lines = NMRDATA_PATH.read_text().splitlines()
    class_start = next(i for i, l in enumerate(nmrdata_lines) if l.strip().startswith("class NMRData"))
    class_body = ast.parse("\n".join(nmrdata_lines[class_start:]))
    class_node = next(n for n in class_body.body if isinstance(n, ast.ClassDef))

    last_method = max(
        (getattr(child, "end_lineno", child.lineno) for child in class_node.body if isinstance(child, ast.FunctionDef)),
        default=0
    )
    class_end = class_start + last_method

    # Remove old stubs if present
    start_marker = "# region Processing stubs"
    end_marker = "# endregion Processing stubs"
    indent = " " * 4

    start_idx = next(i for i, line in enumerate(nmrdata_lines) if line.strip() == start_marker)
    end_idx = next(i for i, line in enumerate(nmrdata_lines) if line.strip() == end_marker)

    if end_idx <= start_idx:
        raise ValueError("Found end marker before start marker")

    # Keep markers and replace contents in between
    new_lines = (
        nmrdata_lines[:start_idx + 1]
        + [indent + "# Auto-generated processing stubs for IDE support"] 
        + [indent + line for line in grouped_stubs if line.strip()]
        + nmrdata_lines[end_idx:]
    )

    NMRDATA_PATH.write_text("\n".join(new_lines))


if __name__ == "__main__":
    inject_alias_docstrings()
    expose_processing_in_init()
    inject_stubs_into_NMRData()