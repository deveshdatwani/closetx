const API = 'http://localhost:8000';
let user = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Popup loaded');
    
    // Restore user session
    chrome.storage.local.get(['user', 'token'], r => {
        if (r.user && r.token) {
            user = r.user;
            document.getElementById('loginView').classList.remove('active');
            document.getElementById('appView').classList.add('active');
            loadImages();
        }
    });

    // Login form listeners
    document.getElementById('loginBtn').addEventListener('click', login);
    document.getElementById('signupLink').addEventListener('click', e => {
        e.preventDefault();
        document.getElementById('signupForm').classList.toggle('hidden');
    });
    document.getElementById('backToLoginLink').addEventListener('click', e => {
        e.preventDefault();
        document.getElementById('signupForm').classList.toggle('hidden');
    });
    document.getElementById('signupBtn').addEventListener('click', signup);

    // App listeners
    document.getElementById('logoutBtn').addEventListener('click', logout);
    document.getElementById('homeTabBtn').addEventListener('click', () => switchTab('home'));
    document.getElementById('uploadTabBtn').addEventListener('click', () => switchTab('upload'));
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
