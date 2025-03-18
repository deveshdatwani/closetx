chrome.action.onClicked.addListener((tab) => {
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: highlightImages
  });
});

function highlightImages() {
  document.querySelectorAll('img').forEach(img => {
    img.style.border = "5px solid blue";
  });
}
