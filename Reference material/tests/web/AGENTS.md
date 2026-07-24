# `tests/web/` — agent guide

Tests for `src/active_inference/web/` (the stdlib HTTP server behind
`run.sh --web`).

```
src/active_inference/web/{server,templates}.py   ↔   tests/web/test_server.py
```

`test_server.py` holds the route / converter / run-endpoint / template /
metadata / favicon / image-helper test classes. The fixture starts the stdlib
HTTP server on an ephemeral localhost port so the tests exercise real request /
response behavior; run-endpoint tests invoke the same subprocess path a user
would trigger from the browser.

## When adding a route or template

1. Add a test class (or extend the matching one) in `test_server.py`.
2. Cover the status code + content type, and assert the rendered HTML contains
   the new element (tab, gallery, button) rather than matching the whole page.

## Tips

- Keep assertions on *structure* (a CSS token, a route prefix, an element id),
  not on exact figure counts — chapter tabs are discovery-driven and grow as
  chapters are added.
- Close `HTTPError` response objects in negative-route tests; warning-free
  pytest treats leaked response sockets as failures.
- The existing CSS/theme-token assertions (e.g. `--accent`, the red dot) pin the
  visual contract; update them deliberately when restyling.
