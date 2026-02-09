// Configuration
const API_BASE_URL = 'http://localhost:8000'; // Change this to your API endpoint

// DOM Elements
const loginView = document.getElementById('loginView');
const mainView = document.getElementById('mainView');
const userInfo = document.getElementById('userInfo');
const username = document.getElementById('username');
const logoutBtn = document.getElementById('logoutBtn');

// Authentication Elements
const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');
const showSignupLink = document.getElementById('showSignup');
const backToLoginLink = document.getElementById('backToLogin');
const signupContainer = document.getElementById('signupContainer');
const signupToggle = document.getElementById('signupToggle');

// Main App Elements
const homeTab = document.getElementById('homeTab');
const uploadTab = document.getElementById('uploadTab');
const imagesGrid = document.getElementById('imagesGrid');
const emptyState = document.getElementById('emptyState');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Upload Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const uploadSuccess = document.getElementById('uploadSuccess');

// Modal Elements
const imageModal = document.getElementById('imageModal');
const modalImage = document.getElementById('modalImage');
const closeModal = document.getElementById('closeModal');
const deleteBtn = document.getElementById('deleteBtn');

// Alert Elements
const errorAlert = document.getElementById('errorAlert');
const successAlert = document.getElementById('successAlert');

let currentUser = null;
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadCurrentUser();
    setupEventListeners();
});

function setupEventListeners() {
    // Auth Events
    loginForm.addEventListener('submit', handleLogin);
    signupForm.addEventListener('submit', handleSignup);
    showSignupLink.addEventListener('click', (e) => {
        e.preventDefault();
        loginView.style.display = 'none';
        signupContainer.style.display = 'block';
    });
    backToLoginLink.addEventListener('click', (e) => {
        e.preventDefault();
        loginView.style.display = 'block';
        signupContainer.style.display = 'none';
    });
    logoutBtn.addEventListener('click', handleLogout);

    // Tab Events
    tabButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tabName = e.target.dataset.tab;
            switchTab(tabName);
        });
    });

    // Upload Events
    uploadBox.addEventListener('click', () => fileInput.click());
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--primary)';
    });
    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = 'var(--border)';
    });
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--border)';
        if (e.dataTransfer.files.length > 0) {
            selectedFile = e.dataTransfer.files[0];
            updateFileInputDisplay();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            updateFileInputDisplay();
        }
    });

    uploadBtn.addEventListener('click', handleUpload);

    // Modal Events
    closeModal.addEventListener('click', () => {
        imageModal.style.display = 'none';
    });
    imageModal.addEventListener('click', (e) => {
        if (e.target === imageModal) {
            imageModal.style.display = 'none';
        }
    });
    deleteBtn.addEventListener('click', handleDelete);
}

function loadCurrentUser() {
    chrome.storage.local.get(['user', 'token'], (result) => {
        if (result.user && result.token) {
            currentUser = result.user;
            showMainView();
        } else {
            showLoginView();
        }
    });
}

function showLoginView() {
    loginView.style.display = 'block';
    mainView.style.display = 'none';
    userInfo.style.display = 'none';
    loginForm.style.display = 'block';
    signupContainer.style.display = 'none';
}

function showMainView() {
    loginView.style.display = 'none';
    mainView.style.display = 'flex';
    userInfo.style.display = 'flex';
    username.textContent = currentUser.username;
    switchTab('home');
    loadImages();
}

function switchTab(tabName) {
    // Update buttons
    tabButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        }
    });

    // Update content
    tabContents.forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName + 'Tab').classList.add('active');

    if (tabName === 'home') {
        loadImages();
    }
}

async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;

    try {
        showAlert('statusAlert', 'Logging in...', 'loading');
        const response = await fetch(`${API_BASE_URL}/user/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ username, password })
        });

        const data = await response.json();

        if (!response.ok) {
            showAlert('errorAlert', data.detail || 'Login failed');
            return;
        }

        // Store user data and token
        currentUser = { id: data.id, username };
        chrome.storage.local.set({ user: currentUser, token: data.access_token }, () => {
            showAlert('successAlert', 'Login successful!');
            loginForm.reset();
            setTimeout(() => showMainView(), 500);
        });
    } catch (error) {
        showAlert('errorAlert', 'Error: ' + error.message);
    }
}

async function handleSignup(e) {
    e.preventDefault();
    const username = document.getElementById('signupUsername').value;
    const password = document.getElementById('signupPassword').value;
    const confirm = document.getElementById('signupConfirm').value;

    if (password !== confirm) {
        showAlert('errorAlert', 'Passwords do not match');
        return;
    }

    try {
        showAlert('statusAlert', 'Creating account...', 'loading');
        const response = await fetch(`${API_BASE_URL}/user/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ username, password })
        });

        const data = await response.json();

        if (!response.ok) {
            showAlert('errorAlert', data.detail || 'Signup failed');
            return;
        }

        // After signup, log in automatically
        const loginResponse = await fetch(`${API_BASE_URL}/user/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ username, password })
        });

        const loginData = await loginResponse.json();

        if (!loginResponse.ok) {
            showAlert('errorAlert', 'Signup successful but login failed. Please log in manually.');
            signupForm.reset();
            backToLoginLink.click();
            return;
        }

        // Store user data
        currentUser = { id: loginData.id, username };
        chrome.storage.local.set({ user: currentUser, token: loginData.access_token }, () => {
            showAlert('successAlert', 'Account created and logged in!');
            signupForm.reset();
            setTimeout(() => showMainView(), 500);
        });
    } catch (error) {
        showAlert('errorAlert', 'Error: ' + error.message);
    }
}

function handleLogout() {
    chrome.storage.local.remove(['user', 'token'], () => {
        currentUser = null;
        showLoginView();
        loginForm.reset();
    });
}

function updateFileInputDisplay() {
    uploadBox.innerHTML = `
        <svg class="upload-icon" viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
        </svg>
        <p>${selectedFile.name}</p>
        <p class="text-muted">${(selectedFile.size / 1024).toFixed(2)} KB</p>
    `;
}

async function handleUpload() {
    if (!selectedFile) {
        showAlert('errorAlert', 'Please select an image first');
        return;
    }

    if (!selectedFile.type.startsWith('image/')) {
        showAlert('errorAlert', 'Please select a valid image file');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('user', currentUser.id);

    uploadBox.style.display = 'none';
    uploadBtn.style.display = 'none';
    uploadProgress.style.display = 'block';

    try {
        const response = await fetch(`${API_BASE_URL}/images/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        progressFill.style.width = '100%';
        progressText.textContent = 'Upload complete!';
        
        setTimeout(() => {
            uploadSuccess.style.display = 'block';
            uploadProgress.style.display = 'none';
            uploadBox.style.display = 'block';
            uploadBtn.style.display = 'block';
            
            // Reset
            selectedFile = null;
            fileInput.value = '';
            uploadBox.innerHTML = `
                <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="16 16 12 12 8 16"></polyline>
                    <line x1="12" y1="12" x2="12" y2="21"></line>
                    <path d="M20.88 18.09A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.29"></path>
                </svg>
                <p>Drag and drop image here</p>
                <p class="text-muted">or click to select</p>
            `;
            
            setTimeout(() => {
                uploadSuccess.style.display = 'none';
                switchTab('home');
            }, 2000);
        }, 500);
    } catch (error) {
        uploadProgress.style.display = 'none';
        uploadBox.style.display = 'block';
        uploadBtn.style.display = 'block';
        showAlert('errorAlert', 'Upload failed: ' + error.message);
    }
}

async function loadImages() {
    if (!currentUser) return;

    imagesGrid.innerHTML = '<div class="loading">Loading your images...</div>';

    try {
        const response = await fetch(`${API_BASE_URL}/images/list?user=${currentUser.id}`);
        const uris = await response.json();

        if (!response.ok) {
            throw new Error('Failed to load images');
        }

        if (uris.length === 0) {
            imagesGrid.innerHTML = '';
            emptyState.style.display = 'block';
            return;
        }

        emptyState.style.display = 'none';
        imagesGrid.innerHTML = '';

        uris.forEach(uri => {
            const imageItem = document.createElement('div');
            imageItem.className = 'image-item';
            imageItem.innerHTML = `<img src="${API_BASE_URL}/images/fetch/${uri}" alt="Garment" data-uri="${uri}">`;
            imageItem.addEventListener('click', () => openImageModal(uri));
            imagesGrid.appendChild(imageItem);
        });
    } catch (error) {
        imagesGrid.innerHTML = '';
        showAlert('errorAlert', 'Failed to load images: ' + error.message);
    }
}

function openImageModal(uri) {
    modalImage.src = `${API_BASE_URL}/images/fetch/${uri}`;
    modalImage.dataset.uri = uri;
    imageModal.style.display = 'flex';
}

async function handleDelete() {
    const uri = modalImage.dataset.uri;
    if (!uri) return;

    if (!confirm('Are you sure you want to delete this image?')) return;

    try {
        const response = await fetch(`${API_BASE_URL}/images/${uri}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Delete failed');
        }

        imageModal.style.display = 'none';
        showAlert('successAlert', 'Image deleted successfully');
        loadImages();
    } catch (error) {
        showAlert('errorAlert', 'Delete failed: ' + error.message);
    }
}

function showAlert(elementId, message, type = 'error') {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.style.display = 'block';

    if (type !== 'loading') {
        setTimeout(() => {
            element.style.display = 'none';
        }, 3000);
    }
}
