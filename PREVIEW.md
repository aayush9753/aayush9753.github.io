# Previewing the al-folio profile (safe, isolated)

This branch (`al-folio`) is a **preview scaffold**. It does **not** touch your
live site (served from `master`) and is **not** auto-deployed anywhere.

## Recommended: Cloudflare Pages on the `al-folio` branch

Gives an isolated `*.pages.dev` URL built from this branch. Your live GitHub
Pages site is untouched (Cloudflare Pages builds separately and serves its own
URL).

Why this path: al-folio's responsive-image plugin (`jekyll-imagemagick`) needs
the ImageMagick binary, which generic CI builders don't have — so it's been
**disabled for the preview** (`imagemagick.enabled: false` in `_config.yml`).
Images still render, just without auto-generated `srcset` sizes.

**Steps (one-time):**
1. Push this branch (ask ALFRED/Claude1 to push, or `git push -u origin al-folio`).
   This only creates a new branch on origin — it does **not** change your live site.
2. Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** →
   **Connect to Git** → pick `aayush9753/aayush9753.github.io`.
3. Set **Production branch = `al-folio`** (NOT master).
4. Build settings:
   - Framework preset: **None**
   - Build command: `bundle exec jekyll build`
   - Build output directory: `_site`
   - (Ruby auto-detected from `.ruby-version` = 3.3.5)
5. Save & Deploy → open the `*.pages.dev` URL.

If the build fails on `bundle install` complaining about a platform, the fix is
`bundle lock --add-platform x86_64-linux` (committed once), or delete
`Gemfile.lock` so Cloudflare resolves gems fresh.

## Higher-fidelity fallback (responsive images): al-folio's own CI

al-folio ships a GitHub Actions workflow that `apt-get install imagemagick`
then builds — full fidelity. It normally deploys to `gh-pages`. To use it for a
preview **without risking your live site**, it must deploy to a *separate*
target (a second Pages project or a `gh-pages-preview` branch served via
Cloudflare Pages), and `imagemagick.enabled` set back to `true`. This is more
setup; only worth it if the `srcset` responsive images matter for the preview.
(Intentionally NOT added here, so nothing auto-deploys when you push.)

## Local preview (your own machine)

If you have Ruby: `bundle install && bundle exec jekyll serve` → http://localhost:4000
