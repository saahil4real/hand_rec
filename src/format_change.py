from PIL import Image
from pdf2image import convert_from_path
import fitz
import cv2
import numpy as np

def convert_to_array(img):
	files = [] 
	# converting pdf to list of np arrays
	if img.lower().endswith('.pdf'):
		# Store Pdf with convert_from_path function
		images = convert_from_path(img)
		for img in images:
			pix = np.array(img)
			files.append(pix)
		# cv2.imshow("result", pix)
		# cv2.waitKey(0)
		return files

	# converting image to numpy array
	else:
		image = cv2.imread(img)
		files.append(image)
		print(type(image))
		print(image.shape)
		print("done")
		return files

# convert_to_png("../data/pdfex.pdf")
# print(res)


 
 
