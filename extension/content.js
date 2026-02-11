// ClosetX content script: sidebar iframe + sticky overlays

// Overlay helpers
function _ensureOverlays() {
  window._closetx_overlays = window._closetx_overlays || []
}

function clearOverlays() {
  _ensureOverlays()
  try {
    window._closetx_overlays.forEach(item => { try { if (item && item.ov) item.ov.remove(); } catch(e){} })
  } catch(e) {}
  window._closetx_overlays = []
  try { window._closetx_last_highlights = []; } catch(e){}
  try { if (window._closetx_overlay_observer) { window._closetx_overlay_observer.disconnect(); window._closetx_overlay_observer = null } } catch(e){}
  try { if (window._closetx_overlay_scroll_fn) { window.removeEventListener('scroll', window._closetx_overlay_scroll_fn, true); window._closetx_overlay_scroll_fn = null } } catch(e){}
  try { if (window._closetx_overlay_resize_fn) { window.removeEventListener('resize', window._closetx_overlay_resize_fn); window._closetx_overlay_resize_fn = null } } catch(e){}
}

function makeOverlayForElement(el, color, label) {
  try {
    const r = el.getBoundingClientRect()
    if (!r.width || !r.height) return null
    const ov = document.createElement('div')
    ov.style.position = 'absolute'
    ov.style.left = (r.left + window.scrollX) + 'px'
    ov.style.top = (r.top + window.scrollY) + 'px'
    ov.style.width = r.width + 'px'
    ov.style.height = r.height + 'px'
    ov.style.border = '4px solid ' + color
    ov.style.boxSizing = 'border-box'
    ov.style.pointerEvents = 'none'
    ov.style.zIndex = '2147483647'
    if (label) {
      const badge = document.createElement('div')
      badge.textContent = label
      badge.style.position = 'absolute'
      badge.style.background = color
      badge.style.color = 'white'
      badge.style.padding = '2px 6px'
      badge.style.fontSize = '12px'
      badge.style.borderRadius = '4px'
      badge.style.left = '4px'
      badge.style.top = '4px'
      ov.appendChild(badge)
    }
    document.documentElement.appendChild(ov)
    _ensureOverlays()
    window._closetx_overlays.push({ ov: ov, el: el })
    _ensureOverlayListeners()
    return ov
  } catch(e) {
    return null
  }
}

function _ensureOverlayListeners() {
  if (!window._closetx_overlay_scroll_fn) {
    window._closetx_overlay_scroll_fn = function() { updateOverlays(); }
    window.addEventListener('scroll', window._closetx_overlay_scroll_fn, true)
  }
  if (!window._closetx_overlay_resize_fn) {
    window._closetx_overlay_resize_fn = function() { updateOverlays(); }
    window.addEventListener('resize', window._closetx_overlay_resize_fn)
  }
  if (!window._closetx_overlay_observer) {
    try {
      window._closetx_overlay_observer = new MutationObserver(function() { updateOverlays(); })
      window._closetx_overlay_observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true })
    } catch(e) { console.warn('MutationObserver not available', e) }
  }
}

function updateOverlays() {
  try {
    if (!window._closetx_overlays || !window._closetx_overlays.length) return
    window._closetx_overlays.forEach(item => {
      try {
        const el = item.el
        const ov = item.ov
        if (!el || !ov) return
        const r = el.getBoundingClientRect()
        ov.style.left = (r.left + window.scrollX) + 'px'
        ov.style.top = (r.top + window.scrollY) + 'px'
        ov.style.width = r.width + 'px'
        ov.style.height = r.height + 'px'
      } catch(e) {}
    })
  } catch(e) {}
}

// Sidebar embedding (separate from overlay logic)
function _ensureSidebar() {
  if (window._closetx_sidebar) return window._closetx_sidebar
  const tab = document.createElement('div')
  tab.id = '_closetx_toggle_tab'
  tab.style.position = 'fixed'
  tab.style.right = '0'
  tab.style.top = '40%'
  tab.style.transform = 'translateY(-50%)'
  tab.style.zIndex = '2147483647'
  tab.style.background = 'linear-gradient(135deg,#6366f1,#ec4899)'
  tab.style.color = 'white'
  tab.style.padding = '8px'
  tab.style.borderTopLeftRadius = '8px'
  tab.style.borderBottomLeftRadius = '8px'
  tab.style.cursor = 'pointer'
  tab.textContent = 'ClosetX'
  document.documentElement.appendChild(tab)

  const iframe = document.createElement('iframe')
  iframe.id = '_closetx_sidebar_iframe'
  iframe.src = chrome.runtime.getURL('popup.html')
  iframe.style.position = 'fixed'
  iframe.style.right = '0'
  iframe.style.top = '0'
  iframe.style.height = '100vh'
  const SIDEBAR_WIDTH = 360
  iframe.style.width = SIDEBAR_WIDTH + 'px'
  iframe.style.border = 'none'
  iframe.style.zIndex = '2147483646'
  iframe.style.boxShadow = '-4px 0 12px rgba(0,0,0,0.2)'
  iframe.style.display = 'none'
  document.documentElement.appendChild(iframe)

  // remember previous margins so we can restore them
  let _prevMargins = null
  tab.addEventListener('click', () => {
    try {
      if (iframe.style.display === 'none') {
        // open: push page content by adding right margin
        _prevMargins = {
          html: document.documentElement.style.marginRight || '',
          body: document.body ? (document.body.style.marginRight || '') : ''
        }
        try { document.documentElement.style.marginRight = SIDEBAR_WIDTH + 'px' } catch(e){}
        try { if (document.body) document.body.style.marginRight = SIDEBAR_WIDTH + 'px' } catch(e){}
        iframe.style.display = 'block'
      } else {
        // close: restore margins
        if (_prevMargins) {
          try { document.documentElement.style.marginRight = _prevMargins.html } catch(e){}
          try { if (document.body) document.body.style.marginRight = _prevMargins.body } catch(e){}
          _prevMargins = null
        }
        iframe.style.display = 'none'
      }
    } catch (e) { console.error(e) }
  })

  window._closetx_sidebar = { tab, iframe }
  return window._closetx_sidebar
}

// Auto-initialize sidebar (do not show iframe until toggled)
try { _ensureSidebar(); } catch(e) { console.error('sidebar init failed', e) }

// Message handling
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  try {
    if (msg && msg.action === 'clearHighlights') { clearOverlays(); sendResponse({ok:true}); return }
    if (msg && msg.action === 'highlightAll') {
      clearOverlays();
      const imgs = Array.from(document.images || []);
      const collected = [];
      imgs.forEach(img => { try { makeOverlayForElement(img, 'rgba(59,130,246,0.9)', 'Image'); const src = img.currentSrc || img.src || ''; if (src) collected.push(src); } catch(e){} });
      try { window._closetx_last_highlights = collected; } catch(e){}
      sendResponse({ok:true, count: collected.length});
      return
    }
    if (msg && msg.action === 'highlight') {
      clearOverlays();
      const urls = msg.urls || [];
      const set = new Set(urls);
      const imgs = Array.from(document.images || []);
      const matched = [];
      imgs.forEach(img => {
        try {
          const src = img.currentSrc || img.src || '';
          if (!src) return;
          for (const u of set) {
            if (!u) continue;
            if (src === u || src.endsWith(u) || src.includes(u)) {
              makeOverlayForElement(img, 'rgba(16,185,129,0.9)', 'Match');
              matched.push(src);
              break
            }
          }
        } catch(e){}
      });
      try { window._closetx_last_highlights = matched; } catch(e){}
      sendResponse({ok:true, matchedCount: matched.length});
      return
    }
    if (msg && msg.action === 'getHighlights') { try { sendResponse({ ok: true, images: window._closetx_last_highlights || [] }); } catch(e) { sendResponse({ ok:false, error: String(e) }) } return }
  } catch (e) { try { sendResponse({ok:false, error:String(e)}) } catch(e){} }
})
