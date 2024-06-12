from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from keras.preprocessing import image
from keras.saving import load_model
#from tensorflow.keras.models import load_model
from notechecker import check
import numpy as np
from ocr import textinfo
import cv2
from process_img import process
import random
import string

app = Flask(__name__)

# Function to process image and run ML models
def process_image(image_path):
    # Example code for ML model prediction
    prediction = 1  # Example prediction, replace with actual ML model prediction
    return prediction


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Read and save the uploaded image using OpenCV
            image1 = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            cv2.imwrite('inputs/input.jpg', image1)
            contour=process('inputs/input.jpg')
            if contour==0:
                return render_template('index.html',note=2)  # Assuming process function saves output as 'outputs/output.jpg'
            processed_image = cv2.imread('outputs/output.jpg')
            check_part1=processed_image[1:139, 847:1425] #(862, 13),(1406, 129) #(847, 1), (1425, 139)
            cv2.imwrite('note_checking/check1.jpg',check_part1)
            note1_result=check('note_checking/check1.jpg',t1=100,t2=200,mode=1)
            check_part2=processed_image[410:552, 1093:1397] #(1093, 410), Bottom-right (1397, 552)
            cv2.imwrite('note_checking/check2.jpg',check_part2)
            note2_result=check('note_checking/check2.jpg',t1=158,t2=240,mode=2)
            if note1_result==0 or note2_result==0 :
                # Example code for dividing image into sub-images based on predefined coordinates
                return render_template('index.html',note=0)
            else:

                predefined_coordinates = [
                    ((952, 530), (1395, 620)),# serial number
                    ((0, 115),(75, 310)), #lines
                    ((470, 9), (665, 175)),#dots
                    ((804, 11),(934, 660)), #thread
                    ((415, 295), (494, 443)), # 500 head
                    ((90, 407), (186, 533)) #register
                ]

                sub_images = [processed_image[y1:y2, x1:x2] for ((x1, y1), (x2, y2)) in predefined_coordinates]
                cv2.imwrite('sub/sub5.jpg',sub_images[4])
                cv2.imwrite('sub/sub6.jpg',sub_images[5])
                # Example code for running ML models on sub-images
                #predictions = [1] * len(sub_images)  # Example predictions, replace with actual ML model predictions
                predictions = []

                model1=load_model('Model1.h5')
                cv2.imwrite('sub/sub1.jpg',sub_images[0])
                img=image.load_img('sub/sub1.jpg',target_size=(144,66))
                x=image.img_to_array(img)
                x=np.expand_dims(x,axis=0)
                images=np.vstack([x])
                val1=model1.predict(images)
                if val1==0:
                    predictions.append(0)
                else:
                    predictions.append(1)

                model2=load_model('Model2.h5')
                cv2.imwrite('sub/sub2.jpg',sub_images[1])
                img=image.load_img('sub/sub2.jpg',target_size=(100,50))
                x=image.img_to_array(img)
                x=np.expand_dims(x,axis=0)
                images=np.vstack([x])
                val2=model2.predict(images)
                if val2==0:
                    predictions.append(0)
                else:
                    predictions.append(1)
            
                model3=load_model('Model3.h5')
                cv2.imwrite('sub/sub3.jpg',sub_images[2])
                original_image = cv2.imread('sub/sub3.jpg', cv2.IMREAD_GRAYSCALE)
                _, binary_image = cv2.threshold(original_image, 190, 210, cv2.THRESH_BINARY) 
                resized_binary_image = cv2.resize(binary_image, (200, 200))
                binary_image_rgb = cv2.cvtColor(resized_binary_image, cv2.COLOR_GRAY2RGB)
                cv2.imwrite('sub/sub7.jpg',binary_image_rgb)
                x=image.img_to_array(binary_image_rgb)
                x=np.expand_dims(x,axis=0)
                images=np.vstack([x])
                val3=model3.predict(images)
                if val3==0:
                    predictions.append(0)
                else:
                    predictions.append(1)

                
                model4=load_model('Model4.h5')
                cv2.imwrite('sub/sub4.jpg',sub_images[3])
                img=image.load_img('sub/sub4.jpg',target_size=(250,60))
                x=image.img_to_array(img)
                x=np.expand_dims(x,axis=0)
                images=np.vstack([x])
                val4=model4.predict(images)
                if val4==0:
                    predictions.append(0)
                else:
                    predictions.append(1)
    
                predictions.append(textinfo('sub/sub5.jpg'))
             
                predictions.append(1)
                final=1
                for i in predictions:
                    final=final*i
                # Mark boundaries based on predictions
                for i, pred in enumerate(predictions):
                    if pred == 1:
                        # Mark green rectangle boundaries
                        cv2.rectangle(processed_image, predefined_coordinates[i][0], predefined_coordinates[i][1], (0, 255, 0), 3)
                    else:
                        # Mark red rectangle boundaries
                        cv2.rectangle(processed_image, predefined_coordinates[i][0], predefined_coordinates[i][1], (0, 0, 255), 3)

                # Save the annotated image
                #randomString = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                #cv2.imwrite(f'app/static/{randomString}.jpg', processed_image)
                cv2.imwrite(f'static/predicted_image.jpg', processed_image)

                return render_template('result.html', prediction=final, image_path='static/predicted_image.jpg')
                #return {"file" : f"{request.base_url}static/{randomString}.jpg", "prediction" : final}

    return render_template('index.html',note=1)

if __name__ == '__main__':
    #app.run(host="0.0.0.0", port="80")
    #app.run(host="10.14.72.216", port="80")
    app.run(debug=True)
