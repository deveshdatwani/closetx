# ClosetX Chrome Extension - Quick Start Guide

## 🚀 What Was Created

A complete Chrome extension for managing your wardrobe with:

- ✅ User authentication (login/signup)
- ✅ Image upload functionality
- ✅ Image gallery/home view
- ✅ Image deletion
- ✅ Modern, responsive UI
- ✅ Local token storage
- ✅ Professional styling

## 📁 Directory Structure

```
closetx/extension/
├── manifest.json              # Extension config
├── popup.html                 # Main UI
├── popup.css                  # Styling
├── popup.js                   # Main logic
├── background.js              # Service worker
├── generate_icons.py          # Icon generator script
├── README.md                  # Full documentation
├── API-CONFIG.md              # API configuration guide
├── QUICKSTART.md              # This file
└── assets/
    ├── icon-16.png            # 16x16 icon ✓
    ├── icon-48.png            # 48x48 icon ✓
    └── icon-128.png           # 128x128 icon ✓
```

## 🔧 Installation (5 minutes)

### Step 1: Configure API Endpoint

Edit `popup.js` line 3:

```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change to your API URL
```

### Step 2: Load in Chrome

1. Open `chrome://extensions/`
2. Toggle "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `closetx/extension` folder
5. Done! ✓

### Step 3: Start Using

Click the ClosetX icon in your Chrome toolbar and:
- Create an account or login
- Upload images of your clothes
- View your wardrobe
- Delete items

## 🎯 Features

### Authentication
- **Login**: Existing users log in with username/password
- **Signup**: New users can create an account
- **Logout**: Clear session and local storage

### Upload
- Drag & drop or click to select images
- File size preview
- Progress indication
- Success confirmation

### Gallery
- Grid view of all uploaded items
- Click to view full size
- Delete option for each item
- Empty state message

## 🔌 API Integration

The extension connects to these endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/user/create` | Register new user |
| POST | `/user/login` | User authentication |
| POST | `/images/upload` | Upload image |
| GET | `/images/list` | Get user's images |
| GET | `/images/fetch/{uri}` | View image |
| DELETE | `/images/{uri}` | Delete image |

## 🧪 Testing

### Quick Test Flow

1. **Sign up**: Create test account
2. **Upload**: Test image upload with a PNG/JPG
3. **Browse**: View image in gallery
4. **Delete**: Remove the test image
5. **Logout**: Clear session

### Test Image Links
- Use any JPG or PNG from your computer
- Recommended size: 300x300px - 2000x2000px
- Max file size: ~10MB

## 🐛 Troubleshooting

### "Failed to connect to API"
```
1. Check API_BASE_URL in popup.js
2. Verify API server is running
3. Check CORS settings on API
4. Review browser console (F12)
```

### "Auth failed"
```
1. Verify username/password
2. Check user exists in database
3. Review API logs
4. Try creating new account
```

### "Upload failed"
```
1. Check file is valid image
2. Ensure user ID is correct
3. Check API storage path exists
4. Review API logs
```

### Icons not showing
```
1. Confirm icons generated (ls assets/)
2. Check permissions on files
3. Reload extension (F5 or refresh)
4. Check console for errors
```

## 📊 Data Flow

```
User Input (popup.js)
    ↓
Chrome Storage API (session data)
    ↓
Fetch API call
    ↓
Backend API
    ↓
Database
```

## 🔐 Security

- Tokens stored in `chrome.storage.local` only
- No persistent credentials
- HTTPS recommended for production
- All requests validated server-side

## 📱 Browser Compatibility

- Chrome 88+
- Edge 88+
- Brave 1.0+
- Other Chromium browsers 88+

## 🎨 Customization

### Change Colors
Edit `popup.css` root variables:

```css
:root {
    --primary: #6366f1;      /* Main color */
    --secondary: #ec4899;    /* Accent color */
    --success: #10b981;      /* Success color */
    --danger: #ef4444;       /* Danger color */
}
```

### Change Size
Edit `body` width in `popup.css`:

```css
body {
    width: 500px;  /* Change this */
    height: 600px; /* Add this */
}
```

### Add New Features
Edit `popup.js` to add:
- New tabs (duplicate tab-btn + tab-content)
- New API calls (follow existing pattern)
- New storage (use `chrome.storage.local`)

## 📝 Next Steps

1. **Test with your API**: Update API_BASE_URL and test
2. **Create test account**: Sign up a test user
3. **Upload test image**: Verify upload works
4. **Generate custom icons**: Edit `generate_icons.py` for custom design
5. **Deploy to Web Store**: Follow Chrome Web Store publishing guide

## 📚 Documentation

- [Full README](./README.md)
- [API Configuration](./API-CONFIG.md)
- [Icon Generator Usage](./generate_icons.py)

## ✨ Pro Tips

1. **Icons**: Edit colors in `generate_icons.py` then re-run it
2. **UI**: Most styling is in `popup.css` root variables
3. **API Calls**: Look at existing fetch examples in `popup.js`
4. **Debugging**: Use DevTools console (Inspect → Console tab)
5. **Hot Reload**: Save file → Reload extension on extensions page

## 🎉 You're all set!

Your Chrome extension is ready to use. Here's what to do next:

```bash
# 1. Make sure your API is running
cd closetx && python api/main.py

# 2. Open Chrome and navigate to
chrome://extensions/

# 3. Click the ClosetX icon and start using!
```

---

**Need help?** Check the console (F12) for detailed error messages!
