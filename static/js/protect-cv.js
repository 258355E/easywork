(() => {
  const blind = () => document.body.classList.add("protect-blind");
  const stop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    blind();
    return false;
  };

  // Deterrents only (not real security).
  document.addEventListener("contextmenu", stop, { capture: true });
  document.addEventListener(
    "dragstart",
    (e) => {
      if (e.target && (e.target.tagName === "IMG" || e.target.closest(".cv-stage"))) stop(e);
    },
    { capture: true }
  );

  document.addEventListener(
    "keydown",
    (e) => {
      const key = (e.key || "").toLowerCase();
      const ctrlOrCmd = e.ctrlKey || e.metaKey;

      // F12 / DevTools shortcuts
      if (key === "f12") return stop(e);
      if (ctrlOrCmd && e.shiftKey && ["i", "j", "c"].includes(key)) return stop(e);
      if (ctrlOrCmd && ["u", "s"].includes(key)) return stop(e);

      // Printing: if download is paywalled, block Ctrl/Cmd+P.
      if (ctrlOrCmd && key === "p" && document.body.dataset.paid === "0") return stop(e);
      return undefined;
    },
    { capture: true }
  );

  // Heuristic devtools detection: if suspected open, blur the CV area.
  const threshold = 160;
  const check = () => {
    const widthGap = Math.abs(window.outerWidth - window.innerWidth);
    const heightGap = Math.abs(window.outerHeight - window.innerHeight);
    if (widthGap > threshold || heightGap > threshold) blind();
  };

  window.addEventListener("resize", check);
  setInterval(check, 1200);
  check();
})();

