import sys
from PIL import Image, ImageDraw

BASE_FOLDER = "./tmp/"
img = Image.new('RGB', (100, 30), color = (73, 109, 137))

d = ImageDraw.Draw(img)
d.text((10,10), "Hello World", fill=(255,255,0))

img.save(BASE_FOLDER + 'teste.jpg')
sys.stdout.write('teste.jpg')
