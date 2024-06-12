import os
import cv2
import numpy as np
from rembg import remove 
from PIL import Image 

def order_points(pts):
	'''Rearrange coordinates to order: 
       top-left, top-right, bottom-right, bottom-left'''
	rect = np.zeros((4, 2), dtype='float32')
	pts = np.array(pts)
	s = pts.sum(axis=1)
	# Top-left point will have the smallest sum.
	rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
	rect[2] = pts[np.argmax(s)]
	
	diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
	rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
	rect[3] = pts[np.argmax(diff)]
	# Return the ordered coordinates.
	return rect.astype('int').tolist()

def process(input_img_path):
    input_img = Image.open(input_img_path)
    orig_img= cv2.imread(input_img_path)
    # Display the input image
    
    # Remove the background
    output_img = remove(input_img)
    
    # Convert to RGB mode
    output_img = output_img.convert("RGB")
    
    
    output_img_path = 'bg/removedbg.jpg'
    output_img.save(output_img_path)
    image = cv2.imread(output_img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # red color boundaries [B, G, R]
    lower = [np.mean(image[:,:,i] - np.std(image[:,:,i])/3 ) for i in range(3)]
    upper = [250, 250, 250]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    ret,thresh = cv2.threshold(mask, 40, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    con = np.zeros_like(image)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    flag=0
    for c in page:
        # Approximate the contour.
        epsilon = 0.1 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points
        if len(corners) == 4:
            flag=1
            break
    # Sorting the corners and converting them to desired shape.
    if flag==0:    
         return 0
         
    else:
        cv2.drawContours(con, c, -1, (0, 255, 255), 3)
        cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
        corners = sorted(np.concatenate(corners).tolist())

        # Displaying the corners.
        for index, c in enumerate(corners):
            character = chr(65 + index)
            cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5, cv2.LINE_AA)

        # Rearranging the order of the corner points.
        corners = order_points(corners)
        (tl, tr, br, bl) = corners
        # Finding the maximum width.
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Finding the maximum height.
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # Final destination co-ordinates.
        destination_corners = [
                [0, 0],
                [maxWidth, 0],
                [maxWidth, maxHeight],
                [0, maxHeight]]
        homography = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
        final = cv2.warpPerspective(orig_img, np.float32(homography), (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    # cv2.imwrite('grabcutop/img22.jpg', final 
        bigger = cv2.resize(final, (1535,665))
        cv2.imwrite(f'outputs/output.jpg', bigger) 
        return 1
    
