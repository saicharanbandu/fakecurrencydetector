import cv2
import json
import requests
from PIL import Image

# Resize the image
def check(input_path,t1,t2,mode):
    image = cv2.imread(input_path)
    cv2.imwrite('note_checking/original_image.jpg',image)
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height
    target_height = 300
    target_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    cv2.imwrite('note_checking/resized_image.jpg', resized_image)

    
    image = cv2.imread('note_checking/resized_image.jpg', cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, t1, t2, cv2.THRESH_BINARY)
    cv2.imwrite('note_checking/binary_image.jpg', binary_image)

    # Perform OCR on the binary image
    img = Image.open('note_checking/binary_image.jpg')
    
    conform1=text_checker('note_checking/original_image.jpg',mode)
    conform2=text_checker('note_checking/binary_image.jpg',mode)

    if conform1==1 or conform2==1:
         return 1
    else:
        return 0



def text_checker(img_path,mode):
    filename=img_path

    payload = {"apikey": "K87417989588957",
            "language": "eng",
            "OCREngine":2,
            }
    with open(filename, 'rb') as f:
        result = requests.post('https://api.ocr.space/parse/image',
                        files={filename: f},
                        data=payload,
                        )
    result = result.content.decode()
    result = json.loads(result)
    parsed_results = result.get("ParsedResults")[0]
    text_detected = parsed_results.get("ParsedText")
    print(text_detected)
    with open('note_checking/abc.txt', mode='w', encoding="utf-8") as file:
            file.write(text_detected)

        # Check if the extracted text indicates a currency note
    with open("note_checking/abc.txt", "r",encoding="utf-8") as file:
        extracted_text = file.read().strip().lower()  # Convert text to lowercase and remove leading/trailing whitespace
        
    if mode==1:
        if "reserve bank of india" in extracted_text or "bank of india" in extracted_text:
            return 1
        else:
            return 0
    if mode==2:
        if "500" in extracted_text:
            return 1
        else:
            return 0
        









        
        
