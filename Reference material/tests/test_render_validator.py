"""Tests for the regenerated-figure validator script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRenderValidator:
    def test_accepts_nonblank_png_and_gif(self, tmp_path: Path) -> None:
        png = tmp_path / "figure.png"
        gif = tmp_path / "anim.gif"
        img = Image.new("RGB", (180, 120), color=(255, 255, 255))
        ImageDraw.Draw(img).rectangle((20, 20, 150, 90), fill=(0, 114, 178))
        img.save(png)
        frames = [Image.new("RGB", (180, 120), color=(255, 255, 255)) for _ in range(2)]
        ImageDraw.Draw(frames[0]).rectangle((20, 20, 120, 80), fill=(0, 114, 178))
        ImageDraw.Draw(frames[1]).ellipse((50, 20, 150, 90), fill=(213, 94, 0))
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)

        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "validate_rendered_figures.py"),
             "--root", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr

    def test_rejects_blank_png(self, tmp_path: Path) -> None:
        blank = tmp_path / "blank.png"
        Image.new("RGB", (180, 120), color=(255, 255, 255)).save(blank)

        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "validate_rendered_figures.py"),
             "--root", str(tmp_path), "--require-variation", "2.0"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "blank.png" in result.stdout
