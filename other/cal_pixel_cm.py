import cv2
import ipdb
from PIL import Image
if __name__ == "__main__":
    image = Image.open("standard.jpg").convert('RGB')
    image = image.resize((720, 540))
    image = image.crop((150,70,574,438))
    image.save("standard_output_2.jpg")