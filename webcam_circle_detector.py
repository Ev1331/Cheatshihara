import numpy as np
import cv2
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

cam = cv2.VideoCapture(1) #0 for the laptop integrated webcam 
ret, frame = cam.read()
h, w, c = frame.shape
#480, 640, 3


def compute(center_x, center_y, radius, img):

    contours = []
    remaining_contours = []

    circle_mask = np.zeros((h, w, 1), dtype=np.uint8) #Black image
    cv2.circle(circle_mask, (w//2-1, h//2-1), h//2, (255, 255, 255), -1) #White circle
    base_mask = np.zeros((h, w), dtype=np.uint8) #Removes everything outside our circle

    # --- Preparation of sampling coordinates ---

    multiplicator = 1 #360/1 samples per circle
    sample_radius_1 = radius - 3
    sample_radius_2 = radius - 15
    sample_points = []

    for angle in range(0, 360//multiplicator):  
        x = int(center_x + sample_radius_1 * np.cos(np.radians(angle*multiplicator)))
        y = int(center_y + sample_radius_1 * np.sin(np.radians(angle*multiplicator)))
        #print(f"center_x:{center_x}, center_y: {center_y}, radius:{radius}, x:{x}, y:{y}")
        sample_points.append((x, y))
        x = int(center_x + sample_radius_2 * np.cos(np.radians(0.5*angle+angle*multiplicator)))
        y = int(center_y + sample_radius_2 * np.sin(np.radians(0.5*angle+angle*multiplicator)))
        sample_points.append((x, y))

    print(f"Computing image...")

    # --- Image pre-processing ---
    
    preprocessed_frame = cv2.convertScaleAbs(img, alpha = 0.4, beta = -120) # a = contrast, b = brightness, gamma = saturation
    preprocessed_frame = cv2.bitwise_not(preprocessed_frame)

    hsv_img = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2HSV)
    base_mask = np.zeros((h, w), dtype=np.uint8)

    # --- Color sampling and masking ---

    for point in sample_points:
        point_color = hsv_img[point[1], point[0]]
        print(f"Color at {point} [H S V]: {point_color}")
        lower_hsv_limit = np.array([point_color[0], 0, 0])
        upper_hsv_limit = np.array([point_color[0], 255, 255])
        hsv_mask = cv2.inRange(hsv_img, lower_hsv_limit, upper_hsv_limit)
        base_mask = cv2.bitwise_or(hsv_mask, base_mask)
        final_mask = cv2.bitwise_not(base_mask)
    
    final_mask = cv2.bitwise_and(final_mask, circle_mask)
    full_mask = final_mask

    # --- Post-processing ---
    
    post_processed_frame = cv2.medianBlur(final_mask, 5) #Remove noise
    post_processed_frame = cv2.dilate(post_processed_frame, np.ones((3, 3), np.uint8), iterations = 5) #Dilate

    # --- Isolate the or the two biggest mask surfaces ---
     
    contours, _ = cv2.findContours(post_processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #Find contours
    largest_contour_1 = max(contours, key=cv2.contourArea) #Take the contour with the largest area
    remaining_contours = [c for c in contours if c is not largest_contour_1] #Remove it from the list
    
    mask_largest_area = np.zeros_like(post_processed_frame) 
    cv2.drawContours(mask_largest_area, [largest_contour_1], -1, 255, thickness=cv2.FILLED) #Redraw largest area on an empty mask

    if len(remaining_contours) != 0: # Are there still contours in the list (i.e. a second digit)?
        largest_contour_2 = max(remaining_contours, key=cv2.contourArea) #Check the second largest
        if cv2.contourArea(largest_contour_2) > 0.3*cv2.contourArea(largest_contour_1): #Is it noise? 
            cv2.drawContours(mask_largest_area, [largest_contour_2], -1, 255, thickness=cv2.FILLED)

    post_processed_frame = cv2.bitwise_and(post_processed_frame, mask_largest_area)

    ocr_image = post_processed_frame
    cv2.imshow("Mask", full_mask)


    # --- OCR and boxes ---

    ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_GRAY2BGR)
   # h, w, c = ocr_image.shape
    boxes = pytesseract.image_to_boxes(ocr_image) 
    for b in boxes.splitlines():
        b = b.split(' ')
        cv2.rectangle(ocr_image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (50, 255, 0), 2)

    myconfig = r'--oem 3 --psm 13'
    print('OCR OUTPUT:')
    ocr_output = pytesseract.image_to_string(ocr_image, config=myconfig)
    ocr_digit = ((ocr_output).split("\n"))[0]
    print(ocr_output)
    #print("digit: ", ocr_digit)
    #print("h,w:", h, w)
    text_y = h-20
    text_x = 20

    filtered_frame = cv2.bitwise_and(cv2.bitwise_not(ocr_image), frame)
    filtered_frame[np.where((filtered_frame == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    if ocr_digit.isdigit():
        filtered_frame = cv2.putText(filtered_frame, f"Result: {ocr_digit}", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    else:
         filtered_frame = cv2.putText(filtered_frame, "Result: ???", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    
    # ---

    cv2.imshow("Character recognition", filtered_frame)

    if cv2.waitKey(0) == 32:
        cv2.destroyWindow("Mask")
        cv2.destroyWindow("Character recognition")
        time.sleep(0.2)

    return img, base_mask


def adjust_gamma(image, gamma=1.0): #I took this (finally unused) function from: https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/
	# build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)



while (True):
    ret, frame = cam.read()
    #frame = adjust_gamma(frame, gamma = 1.6)
    
    """
    #Automatic circle detection (not really helpful)
    h, w, c = frame.shape
    blur = cv2.blur(frame,(15,15))
    blur = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

    circles = cv2.HoughCircles(blur, method=cv2.HOUGH_GRADIENT,dp=0.1,minDist=200, param1=50, param2=12,minRadius = int(h//2-10),maxRadius = int(h//2-5)) 
    if circles is not None: 
        for i in circles[0,:]:
            if (335 < i[0] < 345 and 235 < i[1] < 245):
                i= list(map(int, i))
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                compute(i[0], i[1], i[2], frame)
    """
    
    circle_radius = h//2
    cv2.circle(frame,(w//2, h//2), circle_radius, (0,255,0), 2)
    cv2.imshow('Webcam',frame)

    if cv2.waitKey(5) == 32: #spacebar to snap a picture or exit the current result
        compute(w//2, h//2, circle_radius, frame)

    if cv2.waitKey(5) == 27: #ESC to exit the program
        break    
    
cv2.destroyAllWindows()
cam.release()