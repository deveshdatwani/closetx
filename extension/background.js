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
});

// Keep-alive mechanism for persistent connection (if needed)
setInterval(() => {
    // This keeps the service worker alive
}, 270000); // Every 4.5 minutes
