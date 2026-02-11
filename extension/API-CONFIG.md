<!-- 
API Configuration Guide for ClosetX Extension

This file contains instructions on how to configure the extension to connect to your ClosetX API.
-->

# API Configuration

## Quick Start

1. **Locate the API endpoint** in your ClosetX deployment
   - Local development: `http://localhost:8000`
   - Remote server: Update the URL accordingly

2. **Update popup.js**
   - Open `closetx/extension/popup.js`
   - Find line 3: `const API_BASE_URL = 'http://localhost:8000';`
   - Replace with your API endpoint

### Example Configurations

```javascript
// Local Development
const API_BASE_URL = 'http://localhost:8000';

// Local Network
const API_BASE_URL = 'http://localhost:8000';

// Remote Server (HTTPS recommended)
const API_BASE_URL = 'https://api.closetx.com';

// Docker Compose
const API_BASE_URL = 'http://api:8000'; // May require network config
```

## CORS Configuration

If you get CORS errors, ensure your API server has proper CORS headers:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify extension origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## API Endpoints Reference

All endpoints return JSON responses.

### Authentication

**POST /user/create**
- Create new user account
- Form data: `username`, `password`
- Response: `{ "id": int, "username": str }`

**POST /user/login**
- User login
- Form data: `username`, `password`
- Response: `{ "id": int, "username": str, "access_token": str }`

### Images

**POST /images/upload**
- Upload an image
- Form data: `file` (multipart), `user` (user ID)
- Response: `{ "uri": str }`

**GET /images/list**
- Get list of user's images
- Query params: `user` (user ID)
- Response: Array of URI strings `["uri1", "uri2", ...]`

**GET /images/fetch/{uri}**
- Fetch image by URI
- Path param: `uri` (image URI)
- Response: Image file (PNG/JPG)

**DELETE /images/{uri}**
- Delete an image
- Path param: `uri` (image URI)
- Response: `{ "deleted": str }`

## Testing the API

### Using cURL

```bash
# Create user
curl -X POST http://localhost:8000/user/create \
  -d "username=testuser&password=password123"

# Login
curl -X POST http://localhost:8000/user/login \
  -d "username=testuser&password=password123"

# List images (requires user ID from login)
curl -X GET "http://localhost:8000/images/list?user=1"

# Fetch image
curl -X GET http://localhost:8000/images/fetch/some-uuid > image.png

# Delete image
curl -X DELETE http://localhost:8000/images/some-uuid
```

### Using Python requests

```python
import requests

API_URL = "http://localhost:8000"

# Create user
response = requests.post(f"{API_URL}/user/create", 
    data={"username": "testuser", "password": "pass123"})
print(response.json())

# Login
response = requests.post(f"{API_URL}/user/login",
    data={"username": "testuser", "password": "pass123"})
user_data = response.json()
print(user_data)

# List images
response = requests.get(f"{API_URL}/images/list", 
    params={"user": user_data["id"]})
print(response.json())
```

## Debugging

### Check Console Logs

1. Right-click extension popup → "Inspect"
2. Go to "Console" tab
3. Look for error messages
4. Check Network tab for API request details

### Common Issues

#### CORS Errors
- Extension runs from `chrome-extension://` origin
- API must allow cross-origin requests
- Check browser console for blocked requests

#### 404 Not Found
- Verify API_BASE_URL is correct
- Check API routes are registered
- Ensure API server is running

#### Connection Refused
- Confirm API server is running
- Check firewall rules
- Verify port number is correct

#### 401 Unauthorized
- Token may be expired
- Try logging out and logging back in
- Check if session storage is working

## Security Notes

- Store API tokens securely using `chrome.storage.local`
- Token is only stored locally on user's machine
- Never expose tokens in extension code
- Use HTTPS in production
- Implement rate limiting on API side
- Validate all user inputs on backend

## Production Deployment

Before going to production:

1. **Use HTTPS exclusively**
   ```javascript
   const API_BASE_URL = 'https://api.example.com';
   ```

2. **Enable proper CORS**
   - Specify allowed origins instead of `*`

3. **Implement rate limiting** on server

4. **Add request logging** for monitoring

5. **Test thoroughly** with different network conditions

6. **Update API endpoint** in production code

7. **Publish to Chrome Web Store** (requires review)

## Support

For issues, check:
- Browser console (F12)
- Extension errors (chrome://extensions)
- Backend logs (API server)
- Network requests (DevTools > Network tab)
