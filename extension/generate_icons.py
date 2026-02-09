#!/usr/bin/env python3
"""
Generate placeholder icons for the ClosetX Chrome extension.
Run this script to create icon-16.png, icon-48.png, and icon-128.png
"""

from PIL import Image, ImageDraw
import os

def create_closetx_icon(size):
    """Create a simple ClosetX icon with gradient background and clothing symbol"""
    
    # Create image with gradient
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Draw gradient background
    for y in range(size):
        # Gradient from indigo (#6366f1) to pink (#ec4899)
        r = int(99 + (236 - 99) * (y / size))
        g = int(102 + (72 - 102) * (y / size))
        b = int(241 + (153 - 241) * (y / size))
        draw.rectangle([(0, y), (size, y + 1)], fill=(r, g, b, 255))
    
    # Draw a simple clothing/hanger symbol
    margin = size // 6
    
    # Hanger rod
    draw.line([(margin + size//8, margin), (size - margin - size//8, margin)], 
              fill=(255, 255, 255, 255), width=max(1, size // 32))
    
    # Left side of hanger
    draw.arc([(margin + size//8 - size//16, margin), 
              (margin + size//8 + size//16, margin + size//8)], 
             0, 180, fill=(255, 255, 255, 255), width=max(1, size // 32))
    
    # Right side of hanger
    draw.arc([(size - margin - size//8 - size//16, margin), 
              (size - margin - size//8 + size//16, margin + size//8)], 
             0, 180, fill=(255, 255, 255, 255), width=max(1, size // 32))
    
    # Clothing body (simple rectangle)
    shirt_top = margin + size // 6
    shirt_bottom = size - margin
    shirt_left = margin + size // 8
    shirt_right = size - margin - size // 8
    
    draw.rectangle([(shirt_left, shirt_top), (shirt_right, shirt_bottom)],
                   outline=(255, 255, 255, 255), width=max(1, size // 32))
    
    # Button line down the middle
    draw.line([(size // 2, shirt_top + size // 16), 
               (size // 2, shirt_bottom - size // 16)],
              fill=(255, 255, 255, 255), width=max(1, size // 32))
    
    return img

# Create icons directory if it doesn't exist
icon_dir = os.path.dirname(os.path.abspath(__file__)) + '/assets'
os.makedirs(icon_dir, exist_ok=True)

# Generate icons for different sizes
sizes = [16, 48, 128]

for size in sizes:
    icon = create_closetx_icon(size)
    icon_path = f'{icon_dir}/icon-{size}.png'
    icon.save(icon_path)
    print(f'Created {icon_path}')

print('✓ All icons generated successfully!')
