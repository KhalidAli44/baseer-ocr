from PIL import Image, ImageDraw

# Create white image
img = Image.new('RGB', (800, 600), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Draw vertical line (x=700, width=6, height=600)
draw.rectangle((700, 0, 706, 600), fill=(0, 0, 0))

# Save
img.save("test_images/vertical_line.jpg", "JPEG")
print("Generated: test_images/vertical_line.jpg")