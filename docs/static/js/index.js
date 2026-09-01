(function () {
  'use strict';

  /* ---------- VTP slider + keyframe stepper (ep52) ---------- */
  var NUM_FRAMES = 22;
  var SEQ = 'static/vtp/ep52/kf';

  var compare = document.getElementById('compare');
  var cmpBase = document.getElementById('cmpBase');   // clean frame (no VTP)
  var cmpTop = document.getElementById('cmpTop');      // frame + VTP overlay
  var cmpHandle = document.getElementById('cmpHandle');
  var vpPrev = document.getElementById('vpPrev');
  var vpNext = document.getElementById('vpNext');
  var vpCounter = document.getElementById('vpCounter');

  var pos = 0.88; // start near full +VTP so the 3-view pano reads as one scene
  var idx = 0;   // current keyframe

  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function frameUrl(i, vtp) { return SEQ + pad(i) + (vtp ? '_vtp.jpg' : '.jpg') + '?v=ep52'; }

  var cmpDivider = document.getElementById('cmpDivider');

  if (compare) {
    for (var i = 0; i < NUM_FRAMES; i++) {
      new Image().src = frameUrl(i, false);
      new Image().src = frameUrl(i, true);
    }
  }

  function applySplit() {
    if (!cmpTop || !cmpHandle) return;
    var hideRight = (1 - pos) * 100;
    var clip = 'inset(0 ' + hideRight + '% 0 0)';
    cmpTop.style.clipPath = clip;
    cmpTop.style.webkitClipPath = clip;
    var pct = (pos * 100) + '%';
    cmpHandle.style.left = pct;
    if (cmpDivider) cmpDivider.style.left = pct;
  }
  function setFromClientX(clientX) {
    var rect = compare.getBoundingClientRect();
    if (!rect.width) return;
    pos = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    applySplit();
  }
  function loadFrame(reset) {
    cmpBase.src = frameUrl(idx, false);
    cmpTop.src = frameUrl(idx, true);
    if (vpCounter) vpCounter.textContent = 'Keyframe ' + (idx + 1) + ' / ' + NUM_FRAMES;
    if (vpPrev) vpPrev.disabled = idx === 0;
    if (vpNext) vpNext.disabled = idx === NUM_FRAMES - 1;
    if (reset) pos = 0.88;
    applySplit();
  }
  function step(delta) {
    idx = Math.min(NUM_FRAMES - 1, Math.max(0, idx + delta));
    loadFrame(true);
  }

  var dragging = false;
  if (compare) {
    var start = function (e) {
      dragging = true;
      move(e);
    };
    var end = function () { dragging = false; };
    var move = function (e) {
      if (!dragging) return;
      var x = e.touches ? e.touches[0].clientX : e.clientX;
      setFromClientX(x);
      if (e.cancelable) e.preventDefault();
    };
    compare.addEventListener('mousedown', start);
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', end);
    compare.addEventListener('touchstart', start, { passive: false });
    window.addEventListener('touchmove', move, { passive: false });
    window.addEventListener('touchend', end);

    if (vpPrev) vpPrev.addEventListener('click', function () { step(-1); });
    if (vpNext) vpNext.addEventListener('click', function () { step(1); });

    loadFrame(true);
  }

  /* ---------- copy bibtex ---------- */
  var copyBtn = document.getElementById('copyBib');
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      var text = document.getElementById('bibText').innerText;
      navigator.clipboard.writeText(text).then(function () {
        var old = copyBtn.innerHTML;
        copyBtn.innerHTML = 'Copied';
        setTimeout(function () { copyBtn.innerHTML = old; }, 1600);
      });
    });
  }

  /* ---------- reveal on scroll ---------- */
  var targets = document.querySelectorAll('.section .container > *, .teaser-figure, .contrib-list li');
  targets.forEach(function (el) {
    if (el.id === 'vtpStepper') return;
    el.classList.add('reveal');
  });
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { threshold: 0.08 });
    document.querySelectorAll('.reveal').forEach(function (el) { io.observe(el); });
  } else {
    document.querySelectorAll('.reveal').forEach(function (el) { el.classList.add('in'); });
  }
})();
