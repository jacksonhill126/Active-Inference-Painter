"""Tests for ``utils.notebooks`` — chapter notebook export."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from active_inference.menu.runner import discover_demos, discover_extras, discover_scripts
from active_inference.utils.notebooks import (
    build_notebook,
    default_notebook_dir,
    export_all_notebooks,
    export_chapter_notebook,
    export_demo_notebook,
    export_extra_notebook,
    read_module_docstring,
)


class TestPaths:
    def test_default_notebook_dir_under_output(self) -> None:
        nb = default_notebook_dir()
        assert nb.name == "notebooks"
        assert nb.parent.name == "output"

    def test_default_notebook_dir_honors_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACTIVE_INFERENCE_OUTPUT_ROOT", str(tmp_path / "out"))
        assert default_notebook_dir() == tmp_path / "out" / "notebooks"


class TestDocstringParsing:
    def test_read_module_docstring_first_paragraph(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "chapters"
            / "chapter_01"
            / "01_box_scenario.py"
        )
        doc = read_module_docstring(path)
        assert "box" in doc.lower() or "agent" in doc.lower()


class TestExportChapterNotebook:
    def test_export_chapter_notebook_structure(self, tmp_path: Path) -> None:
        root = tmp_path / "notebooks"
        path = export_chapter_notebook(1, root=root)
        assert path.exists()
        notebook = nbformat.read(path, as_version=4)
        assert notebook.nbformat == 4
        scripts = discover_scripts(1, include_interactive=True)
        assert len(notebook.cells) >= len(scripts) + 2

    def test_script_sections_match_discovery(self, tmp_path: Path) -> None:
        root = tmp_path / "notebooks"
        path = export_chapter_notebook(1, root=root)
        notebook = nbformat.read(path, as_version=4)
        scripts = discover_scripts(1, include_interactive=True)
        for entry in scripts:
            header = f"## `{entry.path.stem}`"
            assert any(
                cell.cell_type == "markdown" and header in cell.source
                for cell in notebook.cells
            )

    def test_animation_section_uses_save_embed(self, tmp_path: Path) -> None:
        root = tmp_path / "notebooks"
        path = export_chapter_notebook(2, root=root)
        notebook = nbformat.read(path, as_version=4)
        code_sources = "".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert "--save" in code_sources
        assert "Image(" in code_sources

    def test_interactive_section_is_markdown_only(self, tmp_path: Path) -> None:
        root = tmp_path / "notebooks"
        path = export_chapter_notebook(1, root=root)
        notebook = nbformat.read(path, as_version=4)
        interactive = [
            entry for entry in discover_scripts(1, include_interactive=True)
            if entry.kind == "interactive"
        ]
        assert interactive
        code_sources = "".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        for entry in interactive:
            assert entry.path.stem not in code_sources or "runpy" not in code_sources


class TestExportBatch:
    def test_export_all_chapters_count(self, tmp_path: Path) -> None:
        root = tmp_path / "notebooks"
        result = export_all_notebooks(
            include_extras=False,
            include_demos=False,
            root=root,
        )
        assert len(result.chapter_paths) == 14
        assert result.total == 14

    def test_export_extra_and_demo(self, tmp_path: Path) -> None:
        root = tmp_path / "notebooks"
        extra_slug = discover_extras()[0].slug
        demo_slug = discover_demos()[0].slug
        extra_path = export_extra_notebook(extra_slug, root=root)
        demo_path = export_demo_notebook(demo_slug, root=root)
        assert extra_path.exists()
        assert demo_path.exists()
        notebook = nbformat.read(extra_path, as_version=4)
        assert notebook.nbformat == 4


class TestBuildNotebook:
    def test_no_animations_skips_run_cells(self) -> None:
        scripts = discover_scripts(2, include_interactive=False)
        notebook = build_notebook(
            scripts,
            title="Chapter 02",
            source_folder="chapters/chapter_02",
            include_animations=False,
        )
        animation_stems = {entry.path.stem for entry in scripts if entry.kind == "animation"}
        for cell in notebook.cells:
            if cell.cell_type == "code":
                for stem in animation_stems:
                    assert stem not in cell.source
