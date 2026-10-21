# Cloudera Design Language (CDL 4.0) alignment

Reference: [Welcome to CDL 4.0](https://design.cloudera.com/9f23dd142/p/62bbf8-welcome-to-cdl-40)

## What “aligned” means here

The shipped UI uses a **CDP-style light console shell** by default (gray page background, white cards, dark sidebar rail, **Cloudera orange** primary actions). When the OS or browser is set to dark mode, tokens switch via `@media (prefers-color-scheme: dark)` in `cdl-tokens.css` (`color-scheme: light | dark` on `:root`). There is no in-app theme toggle yet—preference follows the system. CDL **24px grid gutters** use `--cdl-grid-gutter` / `--cdl-grid-margin`. Legacy hardcoded hex may still appear in specialized views (Ray diagram, cluster map); new work should use tokens only.

Official **CUIX** React components are not bundled (CAI bundle size); we mirror CDL patterns in CSS. For pixel-perfect parity with the design site, paste token values from CDL 4.0 into `cdl-tokens.css` when your team exports them from Figma.

## Tokens

CSS variables live in [`frontend/src/theme/cdl-tokens.css`](../frontend/src/theme/cdl-tokens.css). Global overrides: [`cdl-base.css`](../frontend/src/theme/cdl-base.css).

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
