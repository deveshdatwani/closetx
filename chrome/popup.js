document.addEventListener('DOMContentLoaded', function() {
  // DOM elements
  const closetItems = document.getElementById('closet-items');
  const uploadButton = document.getElementById('upload-button');
  const clearClosetButton = document.getElementById('clear-closet');
  const scanPageButton = document.getElementById('scan-page');
  const matchCount = document.getElementById('match-count');
  const statusMessage = document.getElementById('status-message');
  
  // Load closet items on popup open
  loadClosetItems();
  updateMatchCount();
  
  // Upload button click handler
  uploadButton.addEventListener('click', function() {
    const fileInput = document.getElementById('upload-image');
    
    if (fileInput.files.length === 0) {
      showStatus('Please select an image to upload', 'error');
      return;
    }
    
    const file = fileInput.files[0];
    uploadImage(file);
  });
  
  // Clear closet button click handler
  clearClosetButton.addEventListener('click', function() {
    if (confirm('Are you sure you want to clear your closet?')) {
      chrome.storage.local.remove(['closetItems'], function() {
        loadClosetItems();
        showStatus('Closet cleared successfully', 'success');
        
        // Also notify content script that closet has been cleared
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
          chrome.tabs.sendMessage(tabs[0].id, {action: 'closetCleared'});
        });
      });
    }
  });
  
  // Scan page button click handler
  scanPageButton.addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      chrome.tabs.sendMessage(tabs[0].id, {action: 'scanPage'}, function(response) {
        if (response && response.success) {
          showStatus('Scanning page for matches...', 'success');
        } else {
          showStatus('Error scanning page', 'error');
        }
      });
    });
  });
  
  // Load closet items from storage
  function loadClosetItems() {
    chrome.storage.local.get(['closetItems'], function(result) {
      const items = result.closetItems || [];
      
      if (items.length === 0) {
        closetItems.innerHTML = '<p>No items in your closet yet</p>';
        return;
      }
      
      displayClosetItems(items);
    });
  }
  
  // Display closet items in the UI
  function displayClosetItems(items) {
    closetItems.innerHTML = '';
    
    items.forEach((item, index) => {
      const itemElement = document.createElement('div');
      itemElement.className = 'closet-item';
      itemElement.dataset.index = index;
      
      const img = document.createElement('img');
      img.src = item.image;
      img.alt = 'Closet item';
      
      const colorIndicator = document.createElement('div');
      colorIndicator.className = 'color-indicator';
      colorIndicator.style.backgroundColor = `rgb(${item.color.r}, ${item.color.g}, ${item.color.b})`;
      
      itemElement.appendChild(img);
      itemElement.appendChild(colorIndicator);
      
      // Add event listener to remove item when clicked
      itemElement.addEventListener('click', function() {
        if (confirm('Remove this item from your closet?')) {
          removeClosetItem(index);
        }
      });
      
      closetItems.appendChild(itemElement);
    });
  }
  
  // Upload image to closet
  function uploadImage(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
      const base64Image =e .target.result;
      
      // Simulate color extraction - in a real app, you'd call the color extraction API
      // For demo purposes, we'll use a random color
      const color = {
        r: Math.floor(Math.random() * 256),
        g: Math.floor(Math.random() * 256),
        b: Math.floor(Math.random() * 256)
      };
      
      // Create new closet item
      const newItem = {
        id: Date.now(),
        image: base64Image,
        color: color
      };
      
      // Add to storage
      chrome.storage.local.get(['closetItems'], function(result) {
        const items = result.closetItems || [];
        items.push(newItem);
        
        chrome.storage.local.set({closetItems: items}, function() {
          loadClosetItems();
          showStatus('Item added to closet!', 'success');
          
          // Notify content script that closet has been updated
          chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            chrome.tabs.sendMessage(tabs[0].id, {action: 'closetUpdated'});
          });
        });
      });
    };
    
    reader.readAsDataURL(file);
  }
  
  // Remove item from closet
  function removeClosetItem(index) {
    chrome.storage.local.get(['closetItems'], function(result) {
      const items = result.closetItems || [];
      
      if (index >= 0 && index < items.length) {
        items.splice(index, 1);
        
        chrome.storage.local.set({closetItems: items}, function() {
          loadClosetItems();
          showStatus('Item removed from closet', 'success');
          
          // Notify content script that closet has been updated
          chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            chrome.tabs.sendMessage(tabs[0].id, {action: 'closetUpdated'});
          });
        });
      }
    });
  }
  
  // Update match count display
  function updateMatchCount() {
    chrome.storage.local.get(['matchesFound'], function(result) {
      const matches = result.matchesFound || 0;
      
      if (matches === 0) {
        matchCount.textContent = 'No matches found yet';
      } else {
        matchCount.textContent = `${matches} match${matches === 1 ? '' : 'es'} found`;
      }
    });
  }
  
  // Helper function to show status messages
  function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = type;
    
    // Clear message after 3 seconds
    setTimeout(function() {
      statusMessage.textContent = '';
      statusMessage.className = '';
    }, 3000);
  }
});