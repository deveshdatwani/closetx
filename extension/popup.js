const API = 'http://localhost:8000';
let user = null;

window.addEventListener('error', (e) => {
    try { console.error('Window error', e.error || e.message || e); } catch(_){}
    try { const el = document.getElementById('error'); if (el) { el.textContent = String(e.error || e.message || e); el.classList.add('show'); } } catch(_){}
});
window.addEventListener('unhandledrejection', (ev) => {
    try { console.error('Unhandled rejection', ev.reason); } catch(_){}
    try { const el = document.getElementById('error'); if (el) { el.textContent = String(ev.reason || ev); el.classList.add('show'); } } catch(_){}
});

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('Popup loaded');
        chrome.storage.local.get(['user', 'token'], r => {
            try {
                if (r.user && r.token) {
                    user = r.user;
                    document.getElementById('loginView').classList.remove('active');
                    document.getElementById('appView').classList.add('active');
                    loadImages();
                }
            } catch (e) { console.error('Restore session error', e); }
        });

        try { document.getElementById('loginBtn').addEventListener('click', login); } catch(e){}
        try { document.getElementById('signupLink').addEventListener('click', e => { e.preventDefault(); document.getElementById('signupForm').classList.toggle('hidden'); }); } catch(e){}
        try { document.getElementById('backToLoginLink').addEventListener('click', e => { e.preventDefault(); document.getElementById('signupForm').classList.toggle('hidden'); }); } catch(e){}
        try { document.getElementById('signupBtn').addEventListener('click', signup); } catch(e){}

        try { document.getElementById('logoutBtn').addEventListener('click', logout); } catch(e){}
        try { document.getElementById('matchPageBtn').addEventListener('click', matchPage); } catch(e){}
        try { document.getElementById('highlightAllBtn').addEventListener('click', highlightAll); } catch(e){}
        try { document.getElementById('saveHighlightsBtn').addEventListener('click', saveHighlights); } catch(e){}
        try { document.getElementById('homeTabBtn').addEventListener('click', () => switchTab('home')); } catch(e){}
        try { document.getElementById('uploadTabBtn').addEventListener('click', () => switchTab('upload')); } catch(e){}
    } catch (e) { console.error('DOMContentLoaded handler error', e); }
});

async function login() {
    const u = document.getElementById('username').value;
    const p = document.getElementById('password').value;
    console.log('Login attempt:', u);
    if (!u || !p) return alert('Enter username and password');
    
    try {
        console.log('Fetching from:', `${API}/user/login`);
        const r = await fetch(`${API}/user/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ username: u, password: p })
        });
        console.log('Response status:', r.status);
        const data = await r.json();
        console.log('Login response data:', JSON.stringify(data, null, 2));
        if (!r.ok) throw new Error(data.detail || 'Login failed');
        
        user = data;
        console.log('User object set to:', JSON.stringify(user, null, 2));
        console.log('user.id value:', user.id);
        chrome.storage.local.set({ user: data, token: data.access_token }, () => {
            console.log('Storage saved');
            document.getElementById('loginView').classList.remove('active');
            document.getElementById('appView').classList.add('active');
            loadImages();
        });
    } catch (e) {
        console.error('Login error:', e);
        showError(String(e.message));
    }
}

async function highlightAll() {
    try {
        const tabs = await new Promise(resolve => chrome.tabs.query({ active: true, currentWindow: true }, resolve));
        const tab = tabs && tabs[0];
        if (!tab) return showError('No active tab');
        await sendMessageToTabWithInject(tab.id, { action: 'highlightAll' });
        showSuccess('All images highlighted');
    } catch (e) {
        showError(String(e.message || e));
    }
}

function sendMessageToTabWithInject(tabId, message, timeout = 3000) {
    return new Promise((resolve, reject) => {
        try {
            chrome.tabs.sendMessage(tabId, message, async (resp) => {
                if (chrome.runtime.lastError && /Receiving end does not exist/i.test(chrome.runtime.lastError.message)) {
                    try {
                        await new Promise((res, rej) => {
                            chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] }, (injectionResults) => {
                                if (chrome.runtime.lastError) return rej(chrome.runtime.lastError);
                                res(injectionResults);
                            });
                        });
                    } catch (injErr) {
                        return reject(new Error('Failed to inject content script: ' + String(injErr)));
                    }
                    // retry
                    chrome.tabs.sendMessage(tabId, message, (resp2) => {
                        if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
                        resolve(resp2);
                    });
                    return;
                }
                if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
                resolve(resp);
            });
            // fallback timeout
            setTimeout(() => reject(new Error('sendMessage timeout')), timeout);
        } catch (e) { reject(e); }
    });
}

async function signup() {
    const u = document.getElementById('signupUsername').value;
    const p = document.getElementById('signupPassword').value;
    const c = document.getElementById('signupConfirm').value;
    if (!u || !p || !c) return alert('Fill all fields');
    if (p !== c) return alert('Passwords must match');
    
    try {
        const cr = await fetch(`${API}/user/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ username: u, password: p })
        });
        const cdata = await cr.json();
        if (!cr.ok) throw new Error(cdata.detail || 'Signup failed');
        
        const lr = await fetch(`${API}/user/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ username: u, password: p })
        });
        const ldata = await lr.json();
        user = ldata;
        chrome.storage.local.set({ user: ldata, token: ldata.access_token });
        document.getElementById('loginView').classList.remove('active');
        document.getElementById('appView').classList.add('active');
        loadImages();
    } catch (e) {
        showError(String(e.message));
    }
}

function logout() {
    chrome.storage.local.clear();
    user = null;
    document.getElementById('loginView').classList.add('active');
    document.getElementById('appView').classList.remove('active');
    document.getElementById('username').value = '';
    document.getElementById('password').value = '';
}

function switchTab(tab) {
    // Update button active state
    document.getElementById('homeTabBtn').classList.toggle('active', tab === 'home');
    document.getElementById('uploadTabBtn').classList.toggle('active', tab === 'upload');
    
    // Update tab visibility
    document.getElementById('homeTab').classList.toggle('active', tab === 'home');
    document.getElementById('uploadTab').classList.toggle('active', tab === 'upload');
    
    if (tab === 'home') loadImages();
}

async function loadImages() {
    console.log('loadImages called, user:', user);
    if (!user) {
        console.error('No user set');
        return;
    }
    try {
        console.log('user.id:', user.id);
        const url = `${API}/images/list?user=${user.id}`;
        console.log('Fetching:', url);
        const r = await fetch(url);
        console.log('Status:', r.status);
        const uris = await r.json();
        console.log('Response:', uris);
        
        if (!r.ok) {
            console.error('Request failed:', r.status);
            return;
        }
        
        if (!Array.isArray(uris)) {
            console.error('Not an array:', typeof uris);
            return;
        }
        
        const grid = document.getElementById('imagesList');
        grid.innerHTML = '';
        
        if (uris.length === 0) {
            document.getElementById('emptyMsg').classList.remove('hidden');
            return;
        }
        
        document.getElementById('emptyMsg').classList.add('hidden');
        uris.forEach(uri => {
            const img = document.createElement('div');
            img.className = 'img-item';
            img.style.position = 'relative';
            
            const i = document.createElement('img');
            i.src = `${API}/images/fetch/${uri}`;
            img.appendChild(i);
            
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = '✕';
            deleteBtn.style.position = 'absolute';
            deleteBtn.style.top = '4px';
            deleteBtn.style.right = '4px';
            deleteBtn.style.width = '24px';
            deleteBtn.style.height = '24px';
            deleteBtn.style.padding = '0';
            deleteBtn.style.background = 'rgba(200, 0, 0, 0.8)';
            deleteBtn.style.color = 'white';
            deleteBtn.style.border = 'none';
            deleteBtn.style.borderRadius = '50%';
            deleteBtn.style.cursor = 'pointer';
            deleteBtn.style.fontSize = '16px';
            deleteBtn.style.fontWeight = 'bold';
            deleteBtn.style.display = 'none';
            deleteBtn.addEventListener('mouseenter', () => deleteBtn.style.background = 'rgba(200, 0, 0, 1)');
            deleteBtn.addEventListener('mouseleave', () => deleteBtn.style.background = 'rgba(200, 0, 0, 0.8)');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteImage(uri);
            });
            img.appendChild(deleteBtn);
            
            img.addEventListener('mouseenter', () => deleteBtn.style.display = 'block');
            img.addEventListener('mouseleave', () => deleteBtn.style.display = 'none');
            
            grid.appendChild(img);
        });
        console.log('Loaded', uris.length, 'images');
    } catch (e) {
        console.error('Load error:', e.message);
    }
}

async function deleteImage(uri) {
    if (!confirm('Delete this image?')) return;
    try {
        const r = await fetch(`${API}/images/${uri}`, { method: 'DELETE' });
        if (!r.ok) throw new Error('Delete failed');
        loadImages();
    } catch (e) {
        showError(String(e.message));
    }
}

function showError(msg) {
    const e = document.getElementById('error');
    e.textContent = String(msg).substring(0, 100);
    e.classList.add('show');
    setTimeout(() => e.classList.remove('show'), 3000);
}

function showSuccess(msg) {
    const s = document.getElementById('success');
    s.textContent = String(msg).substring(0, 100);
    s.classList.add('show');
    setTimeout(() => s.classList.remove('show'), 3000);
}

async function matchPage() {
    if (!user) return showError('Login first');
    try {
        const tabs = await new Promise(resolve => chrome.tabs.query({ active: true, currentWindow: true }, resolve));
        const tab = tabs && tabs[0];
        if (!tab) return showError('No active tab');
        let pageImages = [];
        try {
            const exec = await new Promise((resolve, reject) => {
                chrome.scripting.executeScript({ target: { tabId: tab.id }, func: () => {
                    const imgs = Array.from(document.images || []);
                    const urls = imgs.map(i => {
                        const candidates = [i.currentSrc, i.src, i.getAttribute && i.getAttribute('data-src'), i.getAttribute && i.getAttribute('data-lazy')];
                        if (i.srcset) {
                            try { candidates.push(i.srcset.split(',')[0].trim().split(' ')[0]); } catch(e){}
                        }
                        for (const c of candidates) if (c) return c;
                        return null;
                    }).filter(Boolean).filter(u => typeof u === 'string' && u.startsWith('http'));
                    return urls;
                } }, (res) => {
                    if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
                    resolve(res);
                });
            });
            pageImages = (exec && exec[0] && exec[0].result) || [];
        } catch (e) {
            pageImages = [];
        }
        if (!pageImages.length) {
            try {
                const scraped = await new Promise((resolve) => {
                    chrome.runtime.sendMessage({ action: 'scrapeImages', url: tab.url }, (resp) => resolve(resp));
                });
                if (scraped && scraped.ok && Array.isArray(scraped.images)) pageImages = scraped.images;
            } catch (e) {}
        }
        if (!pageImages.length) return showError('No images found on page');
        const resp = await fetch(`${API}/inference/match`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user: user.id, page_images: pageImages }) });
        if (!resp.ok) {
            const d = await resp.json().catch(()=>({detail:'bad response'}));
            return showError(d.detail || 'Match request failed');
        }
        const data = await resp.json();
        const taskId = data.task_id;
        const pageUriMap = data.page_uri_map || {};
        showSuccess('Match queued');
        const results = await pollResult(taskId);
        const highlights = new Set();
        if (results && Array.isArray(results)) {
            // render match pairs sorted by score desc
            results.sort((a,b) => (b.score || 0) - (a.score || 0));
            renderMatches(results, pageUriMap, tab.id);
            results.forEach(r => {
                if (r.highlight) {
                    const src = pageUriMap[r.page_image_url] || r.page_image_url;
                    highlights.add(src);
                }
            });
        }
        chrome.tabs.sendMessage(tab.id, { action: 'highlight', urls: Array.from(highlights) }, () => {});
        showSuccess('Highlights applied');
    } catch (e) {
        showError(String(e.message || e));
    }
}

function renderMatches(results, pageUriMap, tabId) {
    try {
        const container = document.getElementById('matchesList');
        if (!container) return;
        container.innerHTML = '';
        if (!results || !results.length) { container.style.display='none'; return }
        container.style.display = 'block';
        results.forEach(r => {
            const dash = r.dashboard_uri;
            const pageUri = pageUriMap[r.page_image_url] || r.page_image_url;
            const score = (r.score !== undefined) ? r.score : '';
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.gap = '8px';
            row.style.padding = '6px';
            row.style.borderRadius = '6px';
            row.style.background = 'white';
            row.style.marginBottom = '6px';
            row.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)';

            const left = document.createElement('img');
            left.src = `${API}/images/fetch/${dash}`;
            left.style.width = '56px';
            left.style.height = '56px';
            left.style.objectFit = 'cover';
            left.style.borderRadius = '4px';
            left.title = 'Your closet item';
            row.appendChild(left);

            const mid = document.createElement('div');
            mid.style.flex = '1';
            mid.innerHTML = `<div style="font-size:13px;color:#333;margin-bottom:4px;">${pageUri}</div><div style="font-size:12px;color:#666">Score: ${score}</div>`;
            row.appendChild(mid);

            const rightImg = document.createElement('img');
            rightImg.src = pageUri;
            rightImg.style.width = '56px';
            rightImg.style.height = '56px';
            rightImg.style.objectFit = 'cover';
            rightImg.style.borderRadius = '4px';
            rightImg.title = 'Store item';
            row.appendChild(rightImg);

            const btn = document.createElement('button');
            btn.textContent = 'Highlight';
            btn.style.marginLeft = '8px';
            btn.addEventListener('click', async (e) => {
                e.preventDefault();
                try {
                    // tell content script to highlight only this page image
                    await sendMessageToTabWithInject(tabId, { action: 'highlight', urls: [pageUri] });
                    showSuccess('Highlighted on page');
                } catch (err) { showError('Failed to highlight'); }
            });
            row.appendChild(btn);

            container.appendChild(row);
        });
    } catch (e) { console.error('renderMatches', e) }
}

async function pollResult(taskId) {
    const url = `${API}/inference/result/${taskId}`;
    for (let i=0;i<60;i++) {
        try {
            const r = await fetch(url);
            if (!r.ok) { await new Promise(res=>setTimeout(res,1000)); continue; }
            const d = await r.json();
            if (d.result) return d.result;
        } catch (e) {}
        await new Promise(res=>setTimeout(res,1000));
    }
    throw new Error('Timeout waiting for match results');
}

function runtimeSend(msg) {
    return new Promise((resolve, reject) => {
        try {
            chrome.runtime.sendMessage(msg, (resp) => {
                if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
                resolve(resp);
            });
        } catch (e) { reject(e); }
    });
}

async function saveHighlights() {
    if (!user) return showError('Login first');
    try {
        const tabs = await new Promise(resolve => chrome.tabs.query({ active: true, currentWindow: true }, resolve));
        const tab = tabs && tabs[0];
        if (!tab) return showError('No active tab');
        let resp;
        try { resp = await sendMessageToTabWithInject(tab.id, { action: 'getHighlights' }); } catch (e) {
            try { resp = await runtimeSend({ action: 'getHighlights' }); } catch (e2) { resp = null }
        }
        const images = (resp && resp.images) || [];
        if (!images.length) return showError('No highlighted images found');
        showSuccess('Uploading ' + images.length + ' images');
        const uploaded = [];
        for (const url of images) {
            try {
                const fetched = await runtimeSend({ action: 'fetchImage', url });
                if (!fetched || !fetched.ok) { console.warn('fetch failed', url, fetched); continue }
                const dataUrl = `data:${fetched.contentType};base64,${fetched.b64}`;
                const blob = await (await fetch(dataUrl)).blob();
                const form = new FormData();
                form.append('file', blob, url.split('/').pop() || 'image.png');
                const up = await fetch(`${API}/inference/ingest`, { method: 'POST', body: form });
                if (!up.ok) { console.warn('upload failed', url, await up.text()); continue }
                const data = await up.json();
                if (data && data.uri) uploaded.push(data.uri);
            } catch (e) { console.warn('upload error', e); }
        }
        if (uploaded.length) { showSuccess('Saved ' + uploaded.length + ' images'); loadImages(); } else { showError('No images uploaded'); }
    } catch (e) {
        showError(String(e.message || e));
    }
}
