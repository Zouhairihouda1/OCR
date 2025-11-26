import pytesseract
from PIL import Image

img = Image.open("test.png")   # Mets ici le nom de ton image dans /data/input/printed
text = pytesseract.image_to_string(img)

print("---- TEXTE DÉTECTÉ ----")
print(text)
