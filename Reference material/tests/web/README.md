# `tests/web/` — tests for `src/active_inference/web/`

Tests for the stdlib-only local HTTP server behind `run.sh --web`.

| Source file | Test file |
|---|---|
| `web/server.py`, `web/templates.py` | [`test_server.py`](test_server.py) |

## Running

```bash
pytest tests/web -v
```

## What's covered

`test_server.py` exercises:

- **Routes** — each endpoint returns the expected status and content type.
- **Markdown converter** — the inline doc renderer produces the expected HTML.
- **Run endpoint** — the render-figure action dispatches to the right script.
- **Templates** — page HTML embeds the chapter tabs, galleries, and theme tokens.
- **Enriched metadata** — per-script descriptions/figures surface in the UI.
- **Favicon / image helpers** — the embedded assets and image-encoding paths.

These tests start the stdlib server in-process on an ephemeral localhost port,
then use `urllib.request` against real HTTP routes. The run endpoint also uses
the real subprocess dispatch path for cheap representative scripts.
