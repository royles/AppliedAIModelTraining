# Platform version (x.y.z.b)

Single source of truth: **`/VERSION`** (first line matching `^\d+\.\d+\.\d+\.\d+$`).

| Segment | When to increment | Reset |
| --- | --- | --- |
| **x** (major) | Breaking compatibility (API removal, inference format, DB migration users must act on) | y,z,b → 0 |
| **y** (minor) | Net new features (Model registry panel, new training mode) | z,b → 0 |
| **z** (patch) | Small non-breaking changes (docs, defaults, logging, UI copy) | b → 0 |
| **b** (bugfix) | Defect fixes only | — |

**Start:** `1.1.0.0`

The sidebar shows **Platform x.y.z.b**. The hashed `index-*.js` name remains for cache busting only (tooltip).

After editing `VERSION`, rebuild UI when frontend changed:

```bash
python3 scripts/build_ui.py --force
```

`GET /api/ui/build-info` returns `platform_version` and `bundle_js`.
