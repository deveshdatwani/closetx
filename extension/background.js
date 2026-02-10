// Service Worker for ClosetX Extension

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
    if (details.reason === 'install') {
        // Open welcome page or initialize
        console.log('ClosetX extension installed');
    }
});

// Handle messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'checkAuth') {
        chrome.storage.local.get(['user', 'token'], (result) => {
            sendResponse({ 
                authenticated: !!(result.user && result.token),
                user: result.user 
            });
        });
        return true; // Keep channel open for async response
    }
    if (request.action === 'scrapeImages') {
        const url = request.url;
        (async () => {
            try {
                const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Accept': 'text/html' } });
                const text = await res.text();
                const imgs = new Set();
                const imgRe = /<img[^>]+>/gi;
                let m;
                while ((m = imgRe.exec(text)) !== null) {
                    const tag = m[0];
                    const srcMatch = tag.match(/src\s*=\s*"([^"]+)"/i) || tag.match(/src\s*=\s*'([^']+)'/i);
                    const dataSrc = tag.match(/data-src\s*=\s*"([^"]+)"/i) || tag.match(/data-src\s*=\s*'([^']+)'/i);
                    const srcsetMatch = tag.match(/srcset\s*=\s*"([^"]+)"/i) || tag.match(/srcset\s*=\s*'([^']+)'/i);
                    if (srcMatch && srcMatch[1]) imgs.add(new URL(srcMatch[1], url).href);
                    else if (dataSrc && dataSrc[1]) imgs.add(new URL(dataSrc[1], url).href);
                    else if (srcsetMatch && srcsetMatch[1]) {
                        const first = srcsetMatch[1].split(',')[0].trim().split(' ')[0];
                        if (first) imgs.add(new URL(first, url).href);
                    }
                }
                const sourceRe = /<source[^>]+>/gi;
                while ((m = sourceRe.exec(text)) !== null) {
                    const tag = m[0];
                    const srcMatch = tag.match(/srcset\s*=\s*"([^"]+)"/i) || tag.match(/srcset\s*=\s*'([^']+)'/i) || tag.match(/src\s*=\s*"([^"]+)"/i) || tag.match(/src\s*=\s*'([^']+)'/i);
                    if (srcMatch && srcMatch[1]) {
                        const first = srcMatch[1].split(',')[0].trim().split(' ')[0];
                        imgs.add(new URL(first, url).href);
                    }
                }
                const bgRe = /background-image\s*:\s*url\(([^)]+)\)/gi;
                while ((m = bgRe.exec(text)) !== null) {
                    let val = m[1].trim().replace(/^['\"]|['\"]$/g, '');
                    if (val) imgs.add(new URL(val, url).href);
                }
                sendResponse({ ok: true, images: Array.from(imgs) });
            } catch (e) {
                sendResponse({ ok: false, error: String(e) });
            }
        })();
        return true;
    }
    if (request.action === 'fetchImage') {
        (async () => {
            try {
                const res = await fetch(request.url, { headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' } });
                if (!res.ok) return sendResponse({ ok: false, status: res.status });
                const buf = await res.arrayBuffer();
                const bytes = new Uint8Array(buf);
                let binary = '';
                const chunkSize = 0x8000;
                for (let i = 0; i < bytes.length; i += chunkSize) {
                    binary += String.fromCharCode.apply(null, Array.prototype.slice.call(bytes.subarray(i, i + chunkSize)));
                }
                const b64 = btoa(binary);
                const ct = res.headers.get('content-type') || 'application/octet-stream';
                sendResponse({ ok: true, b64, contentType: ct });
            } catch (e) {
                sendResponse({ ok: false, error: String(e) });
            }
        })();
        return true;
    }
});

// Keep-alive mechanism for persistent connection (if needed)
setInterval(() => {
    // This keeps the service worker alive
}, 270000); // Every 4.5 minutes
