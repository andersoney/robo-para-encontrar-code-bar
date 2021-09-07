import cv2
import os
import numpy as np
import pyzbar


class DetectBarCode:
    def __init__(self):
        pass;
    def processImage(self,filename):
        image = cv2.imread(filename)
        scale = 0.3
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # The bigger the kernel, the more the white region increases.
        # If the resizing step was ignored, then the kernel will have to be bigger
        # than the one given here.
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        print(contours);
        for cnt in contours:
            area = cv2.contourArea(cnt)
            xmin, ymin, width, height = cv2.boundingRect(cnt)
            extent = area / (width * height)
            
            # filter non-rectangular objects and small objects
            if (extent > np.pi / 4) and (area > 100):
                bboxes.append((xmin, ymin, xmin + width, ymin + height))
        qrs = []
        info = set()
        for xmin, ymin, xmax, ymax in bboxes:
            roi = image[ymin:ymax, xmin:xmax]
            detections = pyzbar.decode(roi, symbols=[pyzbar.ZBarSymbol.QRCODE])
            for barcode in detections:
                info.add(barcode.data)
                # bounding box coordinates
                x, y, w, h = barcode.rect
                qrs.append((xmin + x, ymin + y, xmin + x + w, ymin + y + height))  


        cv2.imshow("object detection", image)
        # wait until any key is pressed
        cv2.waitKey()

        # save output image to disk
        if(os.path.isdir('detected') == False):
            os.mkdir('detected')
        cv2.imwrite(f"detected/object-detection-single-run.jpg", image)

detectbarcode = DetectBarCode();
detectbarcode.processImage("image.jpg");