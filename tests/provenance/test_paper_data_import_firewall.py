import ast
import json
import subprocess
import sys
import unittest
from importlib.util import resolve_name
from pathlib import Path

import textskill_optimizer.paper as paper


FINAL_MODULE = "textskill_optimizer.paper.final_evaluation"
OPTIMIZATION_MODULES = {
    "textskill_optimizer.paper.backend",
    "textskill_optimizer.paper.data",
    "textskill_optimizer.paper.optimization",
}


def resolved_imports(path: Path, paper_root: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_parent = path.relative_to(paper_root).parent.parts
    package = ".".join(("textskill_optimizer", "paper", *relative_parent))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (
                resolve_name("." * node.level + (node.module or ""), package)
                if node.level
                else node.module or ""
            )
            if node.module:
                imports.append(base)
            imports.extend(
                f"{base}.{alias.name}" for alias in node.names if alias.name != "*"
            )
    return imports


class PaperDataImportFirewallTests(unittest.TestCase):
    def test_cold_final_import_does_not_execute_optimization_package_modules(self) -> None:
        script = """
import json, sys
import textskill_optimizer.paper.final_evaluation
names = [
    name for name in (
        "textskill_optimizer.paper.backend",
        "textskill_optimizer.paper.data",
        "textskill_optimizer.paper.optimization",
    )
    if name in sys.modules
]
print(json.dumps(names))
"""
        completed = subprocess.run(
            (sys.executable, "-c", script),
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertEqual(json.loads(completed.stdout), [])

    def test_final_test_module_and_optimization_modules_are_disconnected(self) -> None:
        self.assertFalse(hasattr(paper, "FinalTestController"))
        paper_root = Path(paper.__file__).parent
        paths = sorted(paper_root.rglob("*.py"))
        modules = {_module_name(path, paper_root): path for path in paths}
        graph: dict[str, set[str]] = {module: set() for module in modules}
        for module, path in modules.items():
            for imported in resolved_imports(path, paper_root):
                target = _owning_module(imported, set(modules))
                if target is not None:
                    graph[module].add(target)

        violations: list[str] = []
        for module in modules:
            reachable = _reachable_modules(module, graph)
            if module != FINAL_MODULE and FINAL_MODULE in reachable:
                violations.append(f"{module} imports final test controller")
            if module == FINAL_MODULE:
                for imported in sorted(reachable & OPTIMIZATION_MODULES):
                    violations.append(f"final test controller imports {imported}")

        self.assertEqual(violations, [])


def _module_name(path: Path, paper_root: Path) -> str:
    module = "textskill_optimizer.paper." + ".".join(
        path.relative_to(paper_root).with_suffix("").parts
    )
    return module.removesuffix(".__init__")


def _owning_module(imported: str, modules: set[str]) -> str | None:
    candidate = imported
    while candidate:
        if candidate in modules:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def _reachable_modules(source: str, graph: dict[str, set[str]]) -> set[str]:
    reached: set[str] = set()
    pending = list(graph[source])
    while pending:
        module = pending.pop()
        if module in reached:
            continue
        reached.add(module)
        pending.extend(graph[module] - reached)
    return reached


if __name__ == "__main__":
    unittest.main()
