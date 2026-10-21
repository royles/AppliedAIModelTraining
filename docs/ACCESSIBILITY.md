# Accessibility (WCAG 2.1 AA target)

## Component rules

- Every form control has a visible `<label>` or `aria-label`.
- Help text uses `FormField` hint + optional `Tooltip` (`aria-describedby`), not `title` alone.
- Icon-only actions use `IconButton` with a required `aria-label`.
- Status changes in the platform bar are announced via `aria-live="polite"`.
- Skip link: **Skip to main content** → `#main-content`.
- Focus: use keyboard Tab; focus ring uses `--cdl-focus-ring` tokens.

## Automated checks

From `frontend/`:

```bash
npm run test:a11y
```

Requires a built UI (`npm run build`) and `@axe-core/cli`. The script scans `dist/index.html` for critical violations.

## Manual smoke

1. Tab through sidebar collapse, nav, wizard steps, and registry list.
2. Zoom to 200% — no clipped primary actions.
3. Enable **prefers-reduced-motion** — status pulse animation should not run.
