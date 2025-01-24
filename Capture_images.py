import os
import cv2
import time
import uuid

IMAGE_PATH='CollectedImages'

labels=['Hello','Yes','No','Thanks','IloveYou','Please']

number_of_images=20

for label in labels:

    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path)
    cap=cv2.VideoCapture(0)

    print('Collecting images for {}'.format(label))
    time.sleep(2)

    if  not cap.isOpened():
        print("Error: Could not open camera.")

    for imgnum in range(number_of_images):
        ret,frame=cap.read()

        if not ret:
            break

        imagename=os.path.join(IMAGE_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        print(imagename)

        cv2.imwrite(imagename,frame)
        cv2.imshow('frame',frame)
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release() 
    
cv2.destroyAllWindows()  