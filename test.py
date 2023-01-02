import cv2

from main import Sample
from main import parser

args = parser()
img = cv2.imread(args.image)

cv2.imwrite('./tmp.jpg', img)
response = Sample.main(args)
print(response)