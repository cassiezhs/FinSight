(function () {
  const SELECTOR = "[data-parallax-depth]";
  const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");

  let targetX = 0;
  let targetY = 0;
  let currentX = 0;
  let currentY = 0;
  let frameId = 0;

  function setDocumentOffset(x, y) {
    document.documentElement.style.setProperty("--parallax-x", x.toFixed(3));
    document.documentElement.style.setProperty("--parallax-y", y.toFixed(3));
  }

  function reset() {
    targetX = 0;
    targetY = 0;
    setDocumentOffset(0, 0);
  }

  function animate() {
    currentX += (targetX - currentX) * 0.075;
    currentY += (targetY - currentY) * 0.075;
    setDocumentOffset(currentX, currentY);
    frameId = requestAnimationFrame(animate);
  }

  function onPointerMove(event) {
    if (REDUCED_MOTION.matches || event.pointerType === "touch") return;
    targetX = event.clientX / window.innerWidth - 0.5;
    targetY = event.clientY / window.innerHeight - 0.5;
  }

  function init() {
    if (REDUCED_MOTION.matches) {
      reset();
      return;
    }

    document.documentElement.classList.add("has-pointer-parallax");
    setDocumentOffset(0, 0);
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerleave", reset, { passive: true });
    frameId = requestAnimationFrame(animate);
  }

  function teardownForReducedMotion() {
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerleave", reset);
    cancelAnimationFrame(frameId);
    document.documentElement.classList.remove("has-pointer-parallax");
    document.querySelectorAll(SELECTOR).forEach((element) => {
      element.style.removeProperty("--parallax-x");
      element.style.removeProperty("--parallax-y");
    });
    reset();
  }

  function onMotionPreferenceChange(event) {
    if (event.matches) {
      teardownForReducedMotion();
    } else {
      init();
    }
  }

  if (typeof REDUCED_MOTION.addEventListener === "function") {
    REDUCED_MOTION.addEventListener("change", onMotionPreferenceChange);
  } else if (typeof REDUCED_MOTION.addListener === "function") {
    REDUCED_MOTION.addListener(onMotionPreferenceChange);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
