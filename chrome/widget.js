// widget.js
// 
document.addEventListener("DOMContentLoaded", () => {
    const imageContainer = document.getElementById("image-container");
  
    // Fetch images from API endpoint
    // fetch("https://api.example.com/images") // Replace with your API endpoint
    //   .then(response => {
    //     if (!response.ok) {
    //       throw new Error(`HTTP error! status: ${response.status}`);
    //     }
    //     return response.json();
    //   })
    //   .then(images => {
    //     images.forEach(imageUrl => {
    //       const img = document.createElement("img");
    //       img.src = imageUrl;
    //       img.className = "api-image";
    //       imageContainer.appendChild(img);
    //     });
    //   })
    //   .catch(error => console.error("Error fetching API images:", error));
  
    // Highlight images on the webpage
    const highlightImages = () => {
      const webpageImages = document.querySelectorAll("img");
      webpageImages.forEach(img => {
        img.style.boxShadow = "0 0 10px 2px #00FF00"; // Apply glow effect
      });
    };
  
    highlightImages();
  
    // Observe changes to dynamically added images
    const observer = new MutationObserver(() => {
      highlightImages();
    });
  
    observer.observe(document.body, { childList: true, subtree: true });
  });