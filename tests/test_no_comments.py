import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_no_comments import violations

assert violations("x = 1\n") == []
assert violations("x = 1  # c\n") == [1]
assert violations("# c\nx = 1\n") == [1]
assert violations('"""doc"""\nx = 1\n') == [1]
assert violations('def f():\n    """doc"""\n    return 1\n') == [2]
assert violations('class A:\n    "doc"\n') == [2]
assert violations('s = "# not a comment"\n') == []
assert violations('f(x="""multi\nline""")\n') == []
print("ok")
