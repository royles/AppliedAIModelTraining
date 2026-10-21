# Cloudera Design Language (CDL 4.0) alignment

Reference: [Welcome to CDL 4.0](https://design.cloudera.com/9f23dd142/p/62bbf8-welcome-to-cdl-40)

## Tokens

CSS variables live in [`frontend/src/theme/cdl-tokens.css`](../frontend/src/theme/cdl-tokens.css):

- Typography: `--cdl-font-sans`, `--cdl-text-*`
- Color: `--cdl-bg-*`, `--cdl-text-*`, `--cdl-primary`, semantic success/warn/error
- Spacing: `--cdl-space-*` (4px grid)
- Focus: `--cdl-focus-ring`, `--cdl-focus-offset`

## Components

| Pattern | Classes |
|---------|---------|
| Primary action | `.btn` |
| Secondary | `.btn-secondary` |
| Danger | `.btn-danger` |
| Surface | `.card` |
| Status | `.pill` |
| Compact | `.btn-compact` |

Shell layout: `.app-root` → status bar → `.app-shell` (sidebar + `main#main-content`).

## Do / don’t

- Do use tokens for new UI; avoid one-off hex colors.
- Do keep contrast ≥ 4.5:1 for body text on surfaces.
- Don’t rely on color alone for status (pair with text in pills and labels).
