"""Validate local links, notebook execution state, and required artifacts."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import nbformat

REPO = Path(__file__).resolve().parents[1]


def check_markdown_links() -> list[str]:
    problems: list[str] = []
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for file in REPO.rglob("*.md"):
        for target in pattern.findall(file.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            local_target = target.split("#", 1)[0]
            if local_target and not (file.parent / local_target).resolve().exists():
                problems.append(f"Missing link in {file.relative_to(REPO)}: {target}")
    return problems


def check_notebooks() -> list[str]:
    problems: list[str] = []
    for file in REPO.rglob("*.ipynb"):
        notebook = nbformat.read(file, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if cell.execution_count is None:
                problems.append(f"Unexecuted code cell {index} in {file.relative_to(REPO)}")
            for output in cell.get("outputs", []):
                if output.output_type == "error":
                    problems.append(
                        f"Error output in {file.relative_to(REPO)} cell {index}: "
                        f"{output.ename}: {output.evalue}"
                    )
    return problems


def main() -> int:
    problems = check_markdown_links() + check_notebooks()
    required = [
        REPO / "data/incident_priority_synthetic.csv",
        REPO / "chapters/01_linear_regression/chapter_01_linear_regression_book.docx",
        REPO / "book/Week_01_ML_Foundations_Study_Guide.docx",
        REPO / "chapters/06_incident_priority_capstone/reports/test_metrics.json",
    ]
    for file in required:
        if not file.exists():
            problems.append(f"Missing required artifact: {file.relative_to(REPO)}")

    if problems:
        print("Repository validation failed:")
        for problem in problems:
            print("-", problem)
        return 1

    print("Repository validation passed")
    print("Markdown files:", len(list(REPO.rglob("*.md"))))
    print("Notebooks:", len(list(REPO.rglob("*.ipynb"))))
    print("PNG visuals:", len(list(REPO.rglob("*.png"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
