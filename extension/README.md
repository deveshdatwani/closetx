# ClosetX Chrome Extension

A modern Chrome extension for managing your wardrobe and fashion items.

## Features

- **User Authentication**: Secure login and signup system
- **Image Upload**: Easily upload images of your clothes and accessories
- **Image Gallery**: View all your uploaded items in a beautiful grid
- **Quick Access**: Access your wardrobe from the Chrome toolbar

## Installation Instructions

### For Development

1. Open `chrome://extensions/` in your Chrome browser
2. Enable "Developer mode" (toggle in the top right)
3. Click "Load unpacked"
4. Select the `closetx/extension` directory
5. The ClosetX extension should now appear in your Chrome toolbar

### Configuration

Before using the extension, update the API endpoint in `popup.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change this to your API endpoint
```

## File Structure

```
extension/
├── manifest.json       # Extension configuration
├── popup.html         # Main UI
├── popup.css          # Styling
├── popup.js           # Main Logic
├── background.js      # Service Worker
├── assets/
│   ├── icon-16.png    # Extension icon (16x16)
│   ├── icon-48.png    # Extension icon (48x48)
│   └── icon-128.png   # Extension icon (128x128)
└── README.md          # This file
```

## Creating Icons

To generate the required icons, you can use:

1. **Online Icon Generator**: Use any PNG/image converter
2. **ImageMagick**: 
   ```bash
   convert source.png -resize 16x16 icon-16.png
   convert source.png -resize 48x48 icon-48.png
   convert source.png -resize 128x128 icon-128.png
   ```
3. **Canva or Figma**: Design custom icons

For now, you can use placeholder images or simple color blocks.

## Usage

### Login/Signup
1. Click the ClosetX icon in your Chrome toolbar
2. Enter your credentials or create a new account
3. Click "Login" or "Sign Up"

### Upload Images
1. Click the "Upload" tab
2. Drag and drop an image or click to select
3. Click "Upload Image"
4. Image will be processed and added to your wardrobe

### View Images
1. Click the "Home" tab (default)
2. All your uploaded items will appear in a grid
3. Click on any image to view it in full
4. Click the delete button to remove an item

## API Endpoints Used

- `POST /user/create` - Create new user account
- `POST /user/login` - User login (returns JWT token and user ID)
- `POST /images/upload` - Upload an image
- `GET /images/list` - Get list of uploaded images (URIs)
- `GET /images/fetch/{uri}` - Fetch image by URI
- `DELETE /images/{uri}` - Delete an image

## Development

### Technologies Used
- Vanilla JavaScript (no frameworks)
- Chrome Extension API
- CSS Grid & Flexbox
- Fetch API for HTTP requests

### Browser Support
- Chrome 88+
- Edge 88+
- Brave 1.0+
- Any Chromium-based browser v88+

## Troubleshooting

### Extension not loading?
- Make sure the manifest.json file exists in the extension directory
- Check the Extensions page for error messages
- Go to chrome://extensions/?errors to see detailed errors

### API connection issues?
- Verify the API_BASE_URL in popup.js matches your backend
- Check CORS settings on your API server
- Ensure the API server is running

### Images not loading?
- Check the API_BASE_URL configuration
- Verify the user ID is correct
- Check browser console for error messages (right-click → Inspect → Console tab)

## Future Features

- [ ] Image categories/tags
- [ ] Search functionality
- [ ] Image filters and effects
- [ ] Sharing wardrobe items
- [ ] Outfit combinations
- [ ] Weather-based outfit recommendations
- [ ] Integration with shopping sites

## License

Proprietary - ClosetX
