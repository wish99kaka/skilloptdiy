import ast
import unittest
from importlib.util import resolve_name
from pathlib import Path

import textskill_optimizer.paper as paper


FORBIDDEN_MODULES = {
    "textskill_optimizer.contract_evidence",
    "textskill_optimizer.contract_rejection_evidence",
    "textskill_optimizer.executive_optimizer",
    "textskill_optimizer.interfaces",
    "textskill_optimizer.models",
    "textskill_optimizer.optimizer",
}


def _resolved_imports(node: ast.ImportFrom, *, package: str) -> list[str]:
    if node.level:
        base = resolve_name("." * node.level + (node.module or ""), package)
    else:
        base = node.module or ""
    imported_names = [
        f"{base}.{alias.name}" for alias in node.names if alias.name != "*"
    ]
    return ([base] if node.module else []) + imported_names


class PaperImportFirewallTests(unittest.TestCase):
    def test_resolves_parent_package_imported_names(self) -> None:
        tree = ast.parse("from .. import executive_optimizer")
        node = next(item for item in ast.walk(tree) if isinstance(item, ast.ImportFrom))

        self.assertEqual(
            _resolved_imports(node, package="textskill_optimizer.paper"),
            ["textskill_optimizer.executive_optimizer"],
        )

    def test_paper_package_does_not_import_extension_semantics(self) -> None:
        paper_root = Path(paper.__file__).parent
        violations: list[str] = []
        for path in sorted(paper_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    relative_parent = path.relative_to(paper_root).parent.parts
                    package = ".".join(
                        ("textskill_optimizer", "paper", *relative_parent)
                    )
                    imports = _resolved_imports(node, package=package)
                else:
                    continue
                for imported in imports:
                    if any(
                        imported == forbidden or imported.startswith(f"{forbidden}.")
                        for forbidden in FORBIDDEN_MODULES
                    ):
                        violations.append(f"{path.relative_to(paper_root)}: {imported}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
