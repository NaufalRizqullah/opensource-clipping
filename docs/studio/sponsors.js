/**
 * Shared Featured Sponsor rendering for OpenSource Clipping Studio.
 * Keeps Settings and Sponsors pages synchronized without markup duplication.
 */
const StudioSponsors = (() => {
  const FEATURED_SPONSOR = {
    bannerSrc: '../assets/sponsors/opensource-clippingxMozhenzhen.png',
    bannerAlt: 'OpenSource Clipping × Mozhenzhen',
    homeUrl: 'https://mozhenzhen.com/go/opensource-clipping',
    ctaUrl: 'https://mozhenzhen.com/go/opensource-clipping',
    features: [
      {
        icon: 'tag',
        accentClass: 'text-emerald-600 dark:text-emerald-400',
        badgeKey: 'sponsors.mozh.priceBadge',
        titleKey: 'sponsors.mozh.priceTitle',
        statsKey: 'sponsors.mozh.priceStats',
        descKey: 'sponsors.mozh.priceDesc',
      },
      {
        icon: 'shield-check',
        accentClass: 'text-sky-600 dark:text-sky-400',
        badgeKey: 'sponsors.mozh.verifyBadge',
        titleKey: 'sponsors.mozh.verifyTitle',
        statsKey: 'sponsors.mozh.verifyStats',
        descKey: 'sponsors.mozh.verifyDesc',
      },
      {
        icon: 'compass',
        accentClass: 'text-amber-600 dark:text-amber-400',
        badgeKey: 'sponsors.mozh.dirBadge',
        titleKey: 'sponsors.mozh.dirTitle',
        statsKey: 'sponsors.mozh.dirStats',
        descKey: 'sponsors.mozh.dirDesc',
      },
      {
        icon: 'gift',
        accentClass: 'text-brand-600 dark:text-brand-400',
        badgeKey: 'sponsors.mozh.perksBadge',
        titleKey: 'sponsors.mozh.perksTitle',
        statsKey: 'sponsors.mozh.perksStats',
        descKey: 'sponsors.mozh.perksDesc',
      },
    ],
  };

  function renderFeatureCard(feature, compact) {
    return `
      <div class="rounded-xl border border-slate-200/80 dark:border-slate-800/80 bg-white/60 dark:bg-slate-900/50 ${compact ? 'p-3' : 'p-4'}">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-1.5">
          <span class="inline-flex items-center gap-1.5 text-xs font-semibold ${feature.accentClass}">
            <i data-lucide="${feature.icon}" class="w-3.5 h-3.5"></i>
            ${StudioI18n.html(feature.badgeKey)}
          </span>
          <span class="text-[11px] font-medium text-slate-500 dark:text-slate-400">${StudioI18n.html(feature.statsKey)}</span>
        </div>
        <p class="text-xs font-semibold text-slate-800 dark:text-slate-100 mb-1">${StudioI18n.html(feature.titleKey)}</p>
        <p class="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">${StudioI18n.html(feature.descKey)}</p>
      </div>
    `;
  }

  /** Render the Featured Sponsor card into a container element or element ID. */
  function renderFeatured(target, options = {}) {
    const container = typeof target === 'string' ? document.getElementById(target) : target;
    if (!container) return;

    const compact = Boolean(options.compact);
    const featureCards = FEATURED_SPONSOR.features
      .map(feature => renderFeatureCard(feature, compact))
      .join('');

    container.innerHTML = `
      <section class="glass-card ${compact ? 'p-5 sm:p-6' : 'p-6 sm:p-7'} rounded-2xl border border-emerald-500/25 dark:border-emerald-500/20 shadow-sm" aria-labelledby="featured-sponsor-heading">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div class="flex flex-wrap items-center gap-2">
            <span class="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 dark:bg-emerald-950/50 dark:text-emerald-300 dark:border-emerald-800/60">
              ${StudioI18n.html('sponsors.featuredBadge')}
            </span>
            <span class="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-semibold bg-amber-50 text-amber-700 border border-amber-200 dark:bg-amber-950/50 dark:text-amber-300 dark:border-amber-800/60">
              ${StudioI18n.html('sponsors.partnerBadge')}
            </span>
          </div>
          ${compact ? `
            <a href="sponsors.html" class="text-xs font-semibold text-brand-600 hover:text-brand-700 dark:text-brand-400 dark:hover:text-brand-300 transition-colors">
              ${StudioI18n.html('sponsors.viewAll')}
            </a>
          ` : ''}
        </div>

        <h3 id="featured-sponsor-heading" class="font-display font-bold ${compact ? 'text-lg' : 'text-xl'} text-slate-900 dark:text-white mb-3">
          ${StudioI18n.html('sponsors.mozh.name')}
        </h3>

        <a href="${FEATURED_SPONSOR.homeUrl}" target="_blank" rel="noopener noreferrer"
           class="block rounded-xl overflow-hidden border border-slate-200/80 dark:border-slate-800/80 mb-4 hover:opacity-95 transition-opacity shadow-sm">
          <img src="${FEATURED_SPONSOR.bannerSrc}" alt="${FEATURED_SPONSOR.bannerAlt}" class="w-full h-auto block" loading="lazy">
        </a>

        <p class="text-sm font-medium text-slate-700 dark:text-slate-200 mb-4 leading-relaxed">
          ${StudioI18n.html('sponsors.mozh.headline')}
        </p>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-5">
          ${featureCards}
        </div>

        <div class="pt-4 border-t border-slate-200/80 dark:border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <p class="text-xs text-slate-500 dark:text-slate-400">${StudioI18n.html('sponsors.mozh.note')}</p>
          <a href="${FEATURED_SPONSOR.ctaUrl}" target="_blank" rel="noopener noreferrer"
             class="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 dark:bg-emerald-500 dark:hover:bg-emerald-600 text-white font-semibold text-xs transition-all shadow-md shadow-emerald-500/20 whitespace-nowrap">
            ${StudioI18n.html('sponsors.mozh.cta')}
            <i data-lucide="external-link" class="w-3.5 h-3.5"></i>
          </a>
        </div>
      </section>
    `;

    if (typeof lucide !== 'undefined' && typeof lucide.createIcons === 'function') {
      lucide.createIcons();
    }
  }

  return { renderFeatured };
})();

