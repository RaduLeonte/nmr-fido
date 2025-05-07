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


if __name__ == "__main__":
    inject_alias_docstrings()
    expose_processing_in_init()