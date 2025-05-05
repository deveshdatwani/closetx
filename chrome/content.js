// Initialize the sidebar
function initClosetXSidebar() {
  // Create the main sidebar container
  const sidebar = document.createElement('div');
  sidebar.id = 'closet-x-sidebar';
  sidebar.className = 'closet-x-sidebar sidebar-collapsed';
  
  // Add the toggle button
  const toggleButton = document.createElement('div');
  toggleButton.className = 'sidebar-toggle';
  toggleButton.innerHTML = 'CX';
  toggleButton.addEventListener('click', toggleSidebar);
  sidebar.appendChild(toggleButton);
  
  // Create sidebar header
  const sidebarHeader = document.createElement('div');
  sidebarHeader.className = 'sidebar-header';
  sidebarHeader.innerHTML = '<h2>ClosetX</h2>';
  sidebar.appendChild(sidebarHeader);
  
  // Create sidebar content
  const sidebarContent = document.createElement('div');
  sidebarContent.className = 'sidebar-content';
  
  // Set up sidebar content
  sidebarContent.innerHTML = `
    <div id="match-status">Scanning for matches...</div>
    <div id="closet-matches"></div>
    
    <div class="my-closet-section">
      <h3>My Closet</h3>
      <div id="closet-items" class="closet-grid">
        <!-- Closet items will be displayed here -->
      </div>
      
      <div class="upload-section">
        <div class="file-input-wrapper">
          <input type="file" id="upload-image" accept="image/*">
        </div>
        <button id="upload-button" class="upload-button">Upload New Item</button>
      </div>
    </div>
  `;
  
  sidebar.appendChild(sidebarContent);
  
  // Create sidebar footer
  const sidebarFooter = document.createElement('div');
  sidebarFooter.className = 'sidebar-footer';
  sidebarFooter.innerHTML = `
    <button id="clear-closet">Clear Closet</button>
    <button id="scan-page">Scan Page</button>
  `;
  sidebar.appendChild(sidebarFooter);
  
  // Add the sidebar to the page
  document.body.appendChild(sidebar);
  
  // Add event listeners for sidebar buttons
  setTimeout(() => {
    const openClosetButton = document.getElementById('clear-closet');
    if (openClosetButton) {
      openClosetButton.addEventListener('click', function() {
        if (confirm('Are you sure you want to clear your closet?')) {
          chrome.storage.local.remove(['closetItems'], function() {
            loadClosetItems();
            updateMatchStatus('Closet cleared successfully');
          });
        }
      });
    }
    
    const scanPageButton = document.getElementById('scan-page');
    if (scanPageButton) {
      scanPageButton.addEventListener('click', function() {
        processProductImages();
        updateMatchStatus('Scanning page for matches...');
      });
    }
    
    const uploadButton = document.getElementById('upload-button');
    if (uploadButton) {
      uploadButton.addEventListener('click', function() {
        const fileInput = document.getElementById('upload-image');
        
        if (fileInput.files.length === 0) {
          updateMatchStatus('Please select an image to upload');
          return;
        }
        
        const file = fileInput.files[0];
        uploadImage(file);
      });
    }
    
    // Load closet items
    loadClosetItems();
  }, 100);
  
  return sidebar;
}

// Toggle sidebar open/close
function toggleSidebar() {
  const sidebar = document.getElementById('closet-x-sidebar');
  if (sidebar) {
    sidebar.classList.toggle('sidebar-collapsed');
  }
}

// Load closet items from storage and display them
function loadClosetItems() {
  const closetItemsContainer = document.getElementById('closet-items');
  if (!closetItemsContainer) return;
  
  chrome.storage.local.get(['closetItems'], function(result) {
    const items = result.closetItems || [];
    
    if (items.length === 0) {
      closetItemsContainer.innerHTML = '<p>No items in your closet yet</p>';
      return;
    }
    
    closetItemsContainer.innerHTML = '';
    
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
      
      closetItemsContainer.appendChild(itemElement);
    });
  });
}

// Upload image to closet
function uploadImage(file) {
  const reader = new FileReader();
  
  reader.onload = function(e) {
    const base64Image = e.target.result;
    
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
        updateMatchStatus('Item added to closet!');
        
        // Refresh matches
        processProductImages();
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
        updateMatchStatus('Item removed from closet');
        
        // Refresh matches
        processProductImages();
      });
    }
  });
}

// Update match status in the sidebar
function updateMatchStatus(message) {
  const matchStatus = document.getElementById('match-status');
  if (matchStatus) {
    matchStatus.textContent = message;
  }
}

// Update the matches display in the sidebar
function updateMatchesDisplay(matchCount) {
  const matchesContainer = document.getElementById('closet-matches');
  if (!matchesContainer) return;
  
  if (matchCount === 0) {
    matchesContainer.innerHTML = '<p>No matches found with your closet</p>';
  } else {
    matchesContainer.innerHTML = `
      <div class="match-notification">
        <p>🎉 Found ${matchCount} match${matchCount === 1 ? '' : 'es'} with your closet!</p>
        <p>Look for highlighted items on the page</p>
      </div>
    `;
  }
  
  // Store match count in storage for popup to access
  chrome.storage.local.set({ matchesFound: matchCount });
}

// Function to get all product images on the page
function getAllProductImages() {
  const productImages = [];
  const images = document.querySelectorAll('img');
  
  images.forEach(img => {
    // Filter for likely product images (could be improved with site-specific selectors)
    if (img.width >= 100 && img.height >= 100 && !img.src.includes('logo') && !img.src.includes('banner')) {
      productImages.push({
        element: img,
        src: img.src
      });
    }
  });
  
  return productImages;
}

// Process product images and check for matches
function processProductImages() {
  const productImages = getAllProductImages();
  if (productImages.length === 0) {
    updateMatchStatus('No product images found on this page');
    return;
  }
  
  updateMatchStatus(`Scanning ${productImages.length} products...`);
  
  // Get closet items from storage
  chrome.storage.local.get(['closetItems'], function(result) {
    const closetItems = result.closetItems || [];
    
    if (closetItems.length === 0) {
      updateMatchStatus('Your closet is empty. Add items to find matches!');
      updateMatchesDisplay(0);
      return;
    }
    
    // Reset any existing highlights
    document.querySelectorAll('.closet-x-match-badge').forEach(el => el.remove());
    document.querySelectorAll('[data-closet-match-id]').forEach(el => {
      el.style.border = '';
      el.removeAttribute('data-closet-match-id');
    });
    
    let matchCount = 0;
    
    // For demo purposes, randomly match some products
    // In a real app, this would use actual color comparison logic
    productImages.forEach(productImage => {
      // Get random closet item index
      const randomIndex = Math.floor(Math.random() * closetItems.length);
      
      // Simulate 30% chance of matching (for demo purposes)
      if (Math.random() < 0.3) {
        const matchingItem = closetItems[randomIndex];
        highlightMatchingProduct(productImage.element, matchingItem.id, matchingItem.color);
        matchCount++;
      }
    });
    
    updateMatchStatus(`Scan complete! Found ${matchCount} matches.`);
    updateMatchesDisplay(matchCount);
  });
}

// Highlight product image with a matching item in closet
function highlightMatchingProduct(imgElement, matchingId, color) {
  // Create a border around the image
  imgElement.style.border = '3px solid #3ee078';
  imgElement.style.borderRadius = '5px';
  
  // Add a badge to indicate a match
  const badge = document.createElement('div');
  badge.className = 'closet-x-match-badge';
  badge.textContent = 'Match!';
  
  // Position the badge properly
  const imgRect = imgElement.getBoundingClientRect();
  const imgParent = imgElement.parentElement;
  
  // Make parent position relative if it's not already
  if (getComputedStyle(imgParent).position === 'static') {
    imgParent.style.position = 'relative';
  }
  
  // Add color indicator to badge if provided
  if (color) {
    const colorDot = document.createElement('span');
    colorDot.className = 'color-dot';
    colorDot.style.backgroundColor = `rgb(${color.r}, ${color.g}, ${color.b})`;
    badge.appendChild(colorDot);
  }
  
  imgParent.appendChild(badge);
  
  // Store matching ID as a data attribute
  imgElement.setAttribute('data-closet-match-id', matchingId);
  
  // Add tooltip showing matched color (optional)
  imgElement.title = 'Matches an item in your closet';
}

// Initialize on page load
let sidebar = null;
window.addEventListener('load', function() {
  // Wait a moment to ensure page is fully loaded
  setTimeout(() => {
    sidebar = initClosetXSidebar();
    processProductImages();
  }, 1500);
});

// Listen for messages from the popup
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'scanPage') {
    processProductImages();
    sendResponse({ success: true });
  } else if (request.action === 'closetUpdated' || request.action === 'closetCleared') {
    // Re-scan page when closet is updated
    processProductImages();
    loadClosetItems();
    sendResponse({ success: true });
  }
  
  // Required to use sendResponse asynchronously
  return true;
});