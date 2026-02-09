# ClosetX Extension - Testing Checklist

Complete this checklist to verify the extension is working correctly.

## Pre-Test Setup

- [ ] API server is running and accessible
- [ ] API endpoint is correct in `popup.js` line 3
- [ ] Chrome extension is loaded in `chrome://extensions/`
- [ ] Extension has no errors (check extensions page)

## Installation & Loading (5 minutes)

- [ ] Extension appears in Chrome toolbar
- [ ] Extension icon displays correctly
- [ ] Can click extension icon to open popup
- [ ] Popup window displays properly (500px width)
- [ ] No console errors when opening popup (F12)

## Authentication Tests

### Signup Flow
- [ ] Click "Sign up" link appears
- [ ] Signup form displays with 3 input fields
- [ ] Can enter username, password, confirm password
- [ ] Password confirmation validation works
- [ ] Error shows if passwords don't match
- [ ] Can successfully create new account
- [ ] Get success message after signup
- [ ] Popup switches to main view automatically

### Login Flow (with existing account)
- [ ] Login form displays on first load
- [ ] Can enter username and password
- [ ] Can successfully login with correct credentials
- [ ] Error message appears for wrong password
- [ ] Error message appears for non-existent user
- [ ] Token is stored after successful login
- [ ] User info shows in header with username
- [ ] Logout button appears in header

### Session Persistence
- [ ] Logged in user stays authenticated after closing popup
- [ ] Refreshing popup keeps user logged in
- [ ] User remains logged in until they click logout
- [ ] Logout clears session and shows login form

## Upload Tests

### UI - Upload Tab
- [ ] "Upload" tab appears and is clickable
- [ ] Upload box displays with icon and text
- [ ] Can click upload box to open file picker
- [ ] File input accepts image files
- [ ] Can drag and drop images onto upload box
- [ ] Upload box changes on hover

### File Selection
- [ ] Selected file name appears in upload box
- [ ] File size displays correctly
- [ ] Only image files can be selected
- [ ] Multiple selection not allowed (single file)

### Upload Process
- [ ] "Upload Image" button is visible
- [ ] Can click button to start upload
- [ ] Progress bar appears during upload
- [ ] Progress percentage updates (0-100%)
- [ ] Success message appears after upload
- [ ] Upload box resets after success
- [ ] Can upload another image immediately

### Upload Error Handling
- [ ] Error message shows if no file selected
- [ ] Error message shows if file is not an image
- [ ] Error message shows if upload fails
- [ ] Can retry upload after error

## Home/Gallery Tests

### Gallery Display
- [ ] "Home" tab appears and is default tab
- [ ] Loading message appears initially
- [ ] Images display in grid layout (3 columns)
- [ ] All uploaded images appear in gallery
- [ ] Images have proper aspect ratio and scaling

### Image Interaction
- [ ] Can click image to view in modal
- [ ] Modal displays full-size image
- [ ] Modal has close button (X)
- [ ] Can close modal by clicking X
- [ ] Can close modal by clicking outside
- [ ] Delete button appears in modal
- [ ] Can delete image from modal
- [ ] Gallery refreshes after delete
- [ ] Deleted image no longer appears

### Empty State
- [ ] "No images yet" message shows for new users
- [ ] Message disappears after uploading image
- [ ] Message reappears after deleting all images

## Tab Navigation

- [ ] Can switch between Home and Upload tabs
- [ ] Active tab is highlighted
- [ ] Clicking tab switches content correctly
- [ ] Tab state is independent

## Error Handling

### API Errors
- [ ] Handles 401 Unauthorized (invalid token)
- [ ] Handles 404 Not Found (missing resource)
- [ ] Handles 500 Server Error gracefully
- [ ] Shows user-friendly error messages
- [ ] No JavaScript errors in console

### Network Errors
- [ ] Handles connection timeout
- [ ] Handles offline mode
- [ ] Handles slow network
- [ ] Retry mechanism available

### User Input Validation
- [ ] Empty username rejected
- [ ] Empty password rejected
- [ ] Username with spaces handled
- [ ] Special characters in password accepted

## Performance Tests

- [ ] Popup loads in < 2 seconds
- [ ] No memory leaks (check DevTools)
- [ ] Smooth animations and transitions
- [ ] No jank or stuttering
- [ ] Image grid scrolls smoothly
- [ ] Upload doesn't freeze UI

### Responsive Design
- [ ] Popup displays at 500px width correctly
- [ ] All text is readable
- [ ] All buttons are clickable
- [ ] No content overflow
- [ ] Forms are properly aligned

## Data Storage Tests

### Local Storage
- [ ] User token is stored
- [ ] User ID is stored
- [ ] Username is stored
- [ ] Storage clears on logout
- [ ] Storage persists on refresh

### Session Data
- [ ] Current user is available
- [ ] Image URIs are correct
- [ ] No sensitive data in plain text

## Browser Compatibility (if applicable)

- [ ] Works in Chrome
- [ ] Works in Edge
- [ ] Works in Brave
- [ ] Works in other Chromium browsers

## Styling & UI Tests

### Colors
- [ ] Header gradient displays correctly
- [ ] Primary button color is visible
- [ ] Text contrast is accessible
- [ ] Hover states are clear

### Fonts
- [ ] All text is readable
- [ ] Font sizes are appropriate
- [ ] Font weight hierarchy is clear

### Spacing
- [ ] Proper padding around elements
- [ ] Proper margins between sections
- [ ] Grid has consistent gaps
- [ ] No overlapping elements

### Icons
- [ ] Extension icon appears in toolbar
- [ ] Icons are clear and visible
- [ ] Icon colors match design

## Console & Debugging

- [ ] No console errors
- [ ] No console warnings
- [ ] All API calls log correctly
- [ ] Error messages are helpful
- [ ] DevTools shows expected network traffic

## Test Scenarios

### Scenario 1: New User Journey
```
1. [ ] Open extension
2. [ ] Create new account
3. [ ] Get logged in automatically
4. [ ] See empty gallery
5. [ ] Upload an image
6. [ ] See image in gallery
7. [ ] Click image to view
8. [ ] Delete the image
9. [ ] See empty message again
```

### Scenario 2: Existing User
```
1. [ ] Open extension
2. [ ] Login with existing account
3. [ ] See previous images
4. [ ] Click image to view
5. [ ] Close modal
6. [ ] Switch to upload tab
7. [ ] Upload new image
8. [ ] Switch to home
9. [ ] See new image in gallery
```

### Scenario 3: Error Recovery
```
1. [ ] Login with wrong password
2. [ ] See error message
3. [ ] Enter correct password
4. [ ] Successfully login
5. [ ] Try to upload non-image file
6. [ ] See error message
7. [ ] Select correct image file
8. [ ] Successfully upload
```

## Final Verification

- [ ] Extension is fully functional
- [ ] No critical issues remain
- [ ] User experience is smooth
- [ ] All features work as intended
- [ ] Ready for production/deployment

---

## Notes

Use this space to document any issues found:

```
Issue: [Describe issue]
Steps to reproduce: [How to replicate]
Expected: [What should happen]
Actual: [What actually happens]
Status: [Open/Fixed]
```

---

## Sign-Off

- [ ] All critical tests passed
- [ ] All major tests passed
- [ ] Minor issues documented
- [ ] Ready for users

**Date Tested**: _______________
**Tested By**: _______________
**Status**: ✓ PASS / ✗ NEEDS WORK

