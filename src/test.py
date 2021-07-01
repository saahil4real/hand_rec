import cv2
import pytesseract
# from format_change import *

def obj_detection(file):
	pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

	# img = cv2.imread("../data/2.jpg")

	# tesseract takes input in rgb
	img = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

	# print(pytesseract.image_to_string(img))
	hImg, wImg,_ = img.shape
	boxes = pytesseract.image_to_data(img)
	results = []
	i=0
	for x, b in enumerate(boxes.splitlines()):
		if x!=0:
			b = b.split()
			print(b)
			if len(b) == 12:
				x,y,w,h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
				# cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
				crop_img = img[y:y+h, x:x+w]
				results.append(crop_img)
				cv2.imwrite("../data/inter/pic" + str(i) + ".png", crop_img)
				i+=1
	return results


# cv2.imshow("result", img)
# cv2.waitKey(0)