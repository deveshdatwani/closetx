document.getElementById('fetch').addEventListener('click', async () => {
    const imagesContainer = document.getElementById('images');
    const errorContainer = document.getElementById('error');
  
    imagesContainer.innerHTML = ''; // Clear previous images
    errorContainer.textContent = ''; // Clear previous errors
  
    try {
      const response = await fetch('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRy0w_Lo5B_DDUybwf69kZEUxfpVXZBRxWJxYitYzgbCNcP4m2j4zqII_tlX2P7SLya6eU&usqp=CAU'); 
      if (!response.ok) throw new Error(`Error: ${response.status}`);
  
      const data = await response.json();
      data.images.forEach(url => {
        const img = document.createElement('img');
        img.src = url;
        imagesContainer.appendChild(img);
      });
    } catch (error) {
      errorContainer.textContent = `Failed to load images: ${error.message}`;
    }
  });