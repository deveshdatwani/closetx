function _ensureOverlays() {
  window._closetx_overlays = window._closetx_overlays || []
}
function clearOverlays() {
  _ensureOverlays()
  window._closetx_overlays.forEach(o => { try { o.remove() } catch(e){} })
  window._closetx_overlays = []
  try { window._closetx_last_highlights = []; } catch(e){}
}
function makeOverlayForElement(el, color, label) {
  const r = el.getBoundingClientRect()
  if (!r.width || !r.height) return null
  const ov = document.createElement('div')
  ov.style.position = 'fixed'
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
  window._closetx_overlays.push(ov)
  return ov
}
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  try {
    if (msg && msg.action === 'clearHighlights') { clearOverlays(); sendResponse({ok:true}); return }
    if (msg && msg.action === 'highlightAll') { clearOverlays(); const imgs = Array.from(document.images || []); const collected = []; imgs.forEach(img => { try { makeOverlayForElement(img, 'rgba(59,130,246,0.9)', 'Image'); const src = img.currentSrc || img.src || ''; if (src) collected.push(src); } catch(e){} }); try { window._closetx_last_highlights = collected; } catch(e){} sendResponse({ok:true, count: collected.length}); return }
      if (msg && msg.action === 'highlight') { clearOverlays(); const urls = msg.urls || []; const set = new Set(urls); const imgs = Array.from(document.images || []); const matched = []; imgs.forEach(img => { try { const src = img.currentSrc || img.src || ''; if (!src) return; for (const u of set) { if (!u) continue; if (src === u || src.endsWith(u) || src.includes(u)) { makeOverlayForElement(img, 'rgba(16,185,129,0.9)', 'Match'); matched.push(src); break } } } catch(e){} }); try { window._closetx_last_highlights = matched; } catch(e){} sendResponse({ok:true, matchedCount: matched.length}); return }
      if (msg && msg.action === 'getHighlights') { try { sendResponse({ ok: true, images: window._closetx_last_highlights || [] }); } catch(e) { sendResponse({ ok:false, error: String(e) }) } return }
  } catch (e) { try { sendResponse({ok:false, error:String(e)}) } catch(e){} }
})
