// Initialize the floating bubble
function initClosetXBubble() {
    // Create the main bubble container
    const bubble = document.createElement('div');
    bubble.id = 'closet-x-bubble';
    bubble.className = 'closet-x-bubble';
    
    // Add the bubble logo/icon
    const bubbleIcon = document.createElement('div');
    bubbleIcon.className = 'bubble-icon';
    bubbleIcon.innerHTML = 'CX';
    bubble.appendChild(bubbleIcon);
    
    // Create the bubble content (initially hidden)
    const bubbleContent = document.createElement('div');
    bubbleContent.className = 'bubble-content';
    bubbleContent.style.display = 'none';
    bubble.appendChild(bubbleContent);
    
    // Set up bubble content
    bubbleContent.innerHTML = `
      <h2>ClosetX</h2>
      <div id="match-status">Scanning for matches...</div>
      <div id="closet-matches"></div>
      <div class="bubble-footer">
        <button id="open-closet">My Closet</button>
        <button id="scan-page">Scan Page</button>
      </div>
    `;
    
    // Add event listeners for bubble buttons
    setTimeout(() => {
      const openClosetButton = document.getElementById('open-closet');
      if (openClosetButton) {
        openClosetButton.addEventListener('click', function() {
          chrome.runtime.sendMessage({ action: 'openPopup' });
        });
      }
      
      const scanPageButton = document.getElementById('scan-page');
      if (scanPageButton) {
        scanPageButton.addEventListener('click', function() {
          processProductImages();
          updateMatchStatus('Scanning page for matches...');
        });
      }
    }, 100);
    
    // Toggle bubble content when icon is clicked
    bubbleIcon.addEventListener('click', function() {
      if (bubbleContent.style.display === 'none') {
        bubbleContent.style.display = 'block';
      } else {
        bubbleContent.style.display = 'none';
      }
    });
    
    // Allow bubble to be draggable
    makeDraggable(bubble);
    
    // Add the bubble to the page
    document.body.appendChild(bubble);
    
    return bubble;
  }
  
  // Make an element draggable
  function makeDraggable(element) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    
    element.querySelector('.bubble-icon').addEventListener('mousedown', dragMouseDown);
    
    function dragMouseDown(e) {
      e.preventDefault();
      // Get cursor position on start
      pos3 = e.clientX;
      pos4 = e.clientY;
      document.addEventListener('mouseup', closeDragElement);
      document.addEventListener('mousemove', elementDrag);
    }
    
    function elementDrag(e) {
      e.preventDefault();
      // Calculate new cursor position
      pos1 = pos3 - e.clientX;
      pos2 = pos4 - e.clientY;
      pos3 = e.clientX;
      pos4 = e.clientY;
      // Set element's new position
      element.style.top = (element.offsetTop - pos2) + "px";
      element.style.left = (element.offsetLeft - pos1) + "px";
    }
    
    function closeDragElement() {
      // Stop moving when mouse button is released
      document.removeEventListener('mouseup', closeDragElement);
      document.removeEventListener('mousemove', elementDrag);
    }
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
  
  // Update match status in the bubble
  function updateMatchStatus(message) {
    const matchStatus = document.getElementById('match-status');
    if (matchStatus) {
      matchStatus.textContent = message;
    }
  }
  
  // Update the matches display in the bubble
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
  
  // Get base64 encoding from image URL
  function getBase64FromImageUrl(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'Anonymous';
      
      img.onload = function() {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        
        try {
          const dataURL = canvas.toDataURL('image/png');
          const base64 = dataURL.split(',')[1];
          resolve(base64);
        } catch (e) {
          // CORS issues might prevent getting base64
          resolve(null);
        }
      };
      
      img.onerror = function() {
        resolve(null);
      };
      
      img.src = url;
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
  let bubble = null;
  window.addEventListener('load', function() {
    // Wait a moment to ensure page is fully loaded
    setTimeout(() => {
      bubble = initClosetXBubble();
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
      sendResponse({ success: true });
    }
    
    // Required to use sendResponse asynchronously
    return true;
  });