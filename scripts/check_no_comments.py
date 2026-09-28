import ast
import io
import json
import sys
import tokenize
from pathlib import Path


def violations(source: str) -> list[int]:
    lines = [
        tok.start[0]
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.COMMENT
    ]
    lines += [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    return sorted(lines)


def notebook_cells(path: Path) -> list[str]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return [
        "".join("\n" if line.lstrip().startswith(("%", "!")) else line for line in cell["source"])
        for cell in cells
        if cell["cell_type"] == "code"
    ]


def main(roots: list[str]) -> int:
    found = 0
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if path.suffix == ".py":
                sources = [path.read_text(encoding="utf-8")]
            elif path.suffix == ".ipynb" and ".ipynb_checkpoints" not in path.parts:
                sources = notebook_cells(path)
            else:
                continue
            for cell, source in enumerate(sources):
                for line in violations(source):
                    where = f"{path}:{line}" if path.suffix == ".py" else f"{path} cell {cell}:{line}"
                    print(f"{where}: комментарий или докстринг — перенеси в .md")
                    found += 1
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["src", "scripts", "tests", "notebooks"]))
