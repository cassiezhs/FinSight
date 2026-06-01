(function () {
  const SELECTOR = ".kpi-item";

  function supportsBackdropFilterUrl() {
    const test = document.createElement("div");
    test.style.backdropFilter = "url(#finsight-liquid-glass-test)";
    return test.style.backdropFilter.indexOf("url") !== -1;
  }

  function svgData(svg) {
    return "data:image/svg+xml;utf8," + encodeURIComponent(svg);
  }

  function displacementMap({ width, height, radius, depth }) {
    const yStart = Math.ceil((radius / height) * 15);
    const yEnd = Math.floor(100 - (radius / height) * 15);
    const xStart = Math.ceil((radius / width) * 15);
    const xEnd = Math.floor(100 - (radius / width) * 15);
    const innerWidth = Math.max(width - depth * 2, 1);
    const innerHeight = Math.max(height - depth * 2, 1);

    return svgData(
      `<svg height="${height}" width="${width}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
        <style>.mix{mix-blend-mode:screen;}</style>
        <defs>
          <linearGradient id="Y" x1="0" x2="0" y1="${yStart}%" y2="${yEnd}%">
            <stop offset="0%" stop-color="#0f0"/>
            <stop offset="100%" stop-color="#000"/>
          </linearGradient>
          <linearGradient id="X" x1="${xStart}%" x2="${xEnd}%" y1="0" y2="0">
            <stop offset="0%" stop-color="#f00"/>
            <stop offset="100%" stop-color="#000"/>
          </linearGradient>
        </defs>
        <rect height="${height}" width="${width}" fill="#808080"/>
        <g filter="blur(2px)">
          <rect height="${height}" width="${width}" fill="#000080"/>
          <rect height="${height}" width="${width}" fill="url(#Y)" class="mix"/>
          <rect height="${height}" width="${width}" fill="url(#X)" class="mix"/>
          <rect x="${depth}" y="${depth}" height="${innerHeight}" width="${innerWidth}" fill="#808080" rx="${radius}" ry="${radius}" filter="blur(${depth}px)"/>
        </g>
      </svg>`
    );
  }

  function displacementFilter({ width, height, radius, depth, strength, chromaticAberration }) {
    const map = displacementMap({ width, height, radius, depth });
    return (
      svgData(
        `<svg height="${height}" width="${width}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="finsight-liquid-displace" color-interpolation-filters="sRGB">
              <feImage x="0" y="0" height="${height}" width="${width}" href="${map}" result="displacementMap"/>
              <feDisplacementMap in="SourceGraphic" in2="displacementMap" scale="${strength + chromaticAberration * 2}" xChannelSelector="R" yChannelSelector="G"/>
              <feColorMatrix type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0" result="displacedR"/>
              <feDisplacementMap in="SourceGraphic" in2="displacementMap" scale="${strength + chromaticAberration}" xChannelSelector="R" yChannelSelector="G"/>
              <feColorMatrix type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0" result="displacedG"/>
              <feDisplacementMap in="SourceGraphic" in2="displacementMap" scale="${strength}" xChannelSelector="R" yChannelSelector="G"/>
              <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0" result="displacedB"/>
              <feBlend in="displacedR" in2="displacedG" mode="screen"/>
              <feBlend in2="displacedB" mode="screen"/>
            </filter>
          </defs>
        </svg>`
      ) + "#finsight-liquid-displace"
    );
  }

  function applyGlass(card) {
    const rect = card.getBoundingClientRect();
    const width = Math.max(Math.round(rect.width), 1);
    const height = Math.max(Math.round(rect.height), 1);
    const radius = parseFloat(getComputedStyle(card).borderRadius) || 16;
    const filter = displacementFilter({
      width,
      height,
      radius,
      depth: 8,
      strength: 54,
      chromaticAberration: 2
    });

    card.style.backdropFilter = `blur(0.5px) url("${filter}") brightness(1.42) saturate(1.34)`;
  }

  function init() {
    if (!supportsBackdropFilterUrl()) {
      document.documentElement.classList.add("liquid-glass-fallback");
      return;
    }

    document.documentElement.classList.add("liquid-glass-displacement");
    const watchedCards = new WeakSet();
    const observer = new ResizeObserver((entries) => {
      entries.forEach((entry) => applyGlass(entry.target));
    });

    function watchCards(root) {
      root.querySelectorAll(SELECTOR).forEach((card) => {
        if (watchedCards.has(card)) return;
        watchedCards.add(card);
        applyGlass(card);
        observer.observe(card);
      });
    }

    watchCards(document);
    const dashObserver = new MutationObserver((entries) => {
      entries.forEach((entry) => {
        entry.addedNodes.forEach((node) => {
          if (!(node instanceof Element)) return;
          if (node.matches(SELECTOR)) {
            watchCards(node.parentElement || document);
          } else {
            watchCards(node);
          }
        });
      });
    });

    dashObserver.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
