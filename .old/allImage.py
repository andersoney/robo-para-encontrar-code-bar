import dozero
import argparse

ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

images = [
    'images\dump1.jpg', 
    'images\dump2.jpg', 
    'images\dump3.jpg', 
    'images\dump4.jfif', 
    'images\dump5.jpg',
    'images\dump6.jpg'
]
findObjectInImage = dozero.FindObjectInImage(args)
for image in images:
    findObjectInImage.processImage(image, False)
