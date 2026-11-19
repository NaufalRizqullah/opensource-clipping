# Studio interface languages

The GitHub Pages interface in `docs/studio/` supports English (`en`) and Indonesian (`id`). New visitors always see English, regardless of their browser language. The **Language** selector is in every page's sidebar, including the mobile navigation menu, and works without a backend connection.

An explicit selection is saved in `localStorage` under `osc_studio_language`. Missing or unsupported preferences fall back to English. If storage is unavailable, switching still works on the current page. Changing languages does not reload the page, clear form values, submit settings, restart progress subscriptions, or replace video players.

This preference controls UI labels, help text, placeholders, accessibility labels, client validation/errors, job statuses, and dates. Backend status codes and request payloads stay unchanged. Source URLs, model aliases, generated clip titles, subtitles, raw server errors, and activity/progress logs retain their original values. This change applies to the GitHub Pages Studio, not the separate React dashboard under `web/dashboard/`.

## Adding or editing translations

`studio/i18n.js` contains both catalogs and the shared `StudioI18n` helpers. It loads before `studio/api.js` on all four pages. No build step or translation service is required.

- Add each key to both `en` and `id`, using the same named `{parameters}`.
- Mark static text with `data-i18n="key"` on a text-only element. Keep an English fallback in the HTML. Wrap only the text when a button or heading also contains an icon or another control.
- Translate attributes with `data-i18n-placeholder`, `data-i18n-title`, `data-i18n-aria-label`, or `data-i18n-content`.
- Use `StudioI18n.text(element, key, params)` for dynamic text and `StudioI18n.html(key, params)` for text inside HTML templates. The latter escapes text and parameters and retains bindings for subsequent language changes.
- Use `StudioI18n.literal(element, value)` when replacing a localized fallback with raw backend content, so later language changes cannot overwrite it.
- Use `StudioI18n.error(key, params)` for client-generated errors and `StudioI18n.errorMessage(error)` when displaying them. Raw backend messages remain verbatim.
- Use `data-i18n-date` for timestamps that must update when the language changes. Dates use `en-US` or `id-ID` formatting in the browser's time zone.

Avoid rebuilding forms or media containers on `studio:language`. The shared translator updates only the marked text and attributes, leaving user input and playback state intact.

## Verification

Run the dependency-free Studio tests with Node.js 20 or newer:

```sh
node --test tests/studio.test.cjs
```

For browser checks, serve `docs/` locally and visit `/studio/`. Check both languages on all four pages, navigation/reload persistence, the connection dialog, mobile menu, light/dark themes, and language changes with unsaved form values or job progress visible. Verify that generated clip titles and raw backend logs remain unchanged.
