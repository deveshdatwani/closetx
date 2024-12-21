document.getElementById('highlightButton').addEventListener('click', () => {
    chrome.scripting.executeScript({
        target: { tabId: chrome.tabs.TAB_ID },
        func: highlightImages
    });
});

function highlightImages() {
    document.querySelectorAll('img').forEach(img => {
        img.style.border = '5px solid red';
    });
}