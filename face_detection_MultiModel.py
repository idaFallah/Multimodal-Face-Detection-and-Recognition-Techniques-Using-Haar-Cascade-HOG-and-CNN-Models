
import cv2

from google.colab import drive
drive.mount('/content/drive')

image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/people1.jpg')

image.shape

from google.colab.patches import cv2_imshow

cv2_imshow(image)

image = cv2.resize(image, (800, 600))

image.shape

cv2_imshow(image)

600 * 800 * 3, 600 * 800, 1440000 - 480000 # difference between pixels

# since the difference is big and in cascade alg it's recommended to use grayscale pics, we monotone the image
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(img_gray)

image.shape, img_gray.shape

# detecting faces with cascade classifier
face_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Computer Vision/Cascades/haarcascade_frontalface_default.xml')

detections = face_detector.detectMultiScale(img_gray)

detections # each row in this matrix indicates a face in the image that was detected
# the first two values in each row are the x and y of where the face is(position)
# the next two values in each row are the sizes of the faces

len(detections)

for (x, y, w, h) in detections:
  print(x, y, w, h)

# drawing rectangle around faces
for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2_imshow(image)

# haarcascade parameters -> to get better results
image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/people1.jpg')
image = cv2.resize(image,(800, 600))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.075) # scale factor is the haarcascade param here
# it starts from 1
# it's correlated to the sizes of the objects(faces) we wanna detect
# if face = smal then add a little bit to 1
# if face = big add more to 1
# until u get the right value for scale factor
for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
cv2_imshow(image)

# another image

# haarcascade parameters -> to get better results
image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/people2.jpg')
# image = cv2.resize(image,(800, 600)),,,,,, the size of image is okay no need for resizing
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=7,
                                            minSize=(20, 20), maxSize=(100, 100))
# when scale factor is not enough t improve results -> min neighbors
# min neighbors = number of candidate neigbors that must exist in rectangles -> for each final binding box
# min size = min size of faces
# max size = max size of faces
for (x, y, w, h) in detections:
  print(w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2_imshow(image)

# we need stronger models for the remaining faces

# eye detection

image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/people1.jpg')
image = cv2.resize(image,(800, 600))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.075)
for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2_imshow(image)

eye_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Computer Vision/Cascades/haarcascade_eye.xml')

image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/people1.jpg')
#image = cv2.resize(image,(800, 600)),,,,,,,,, size of the image is also important in improving the results, for example this image is actually a big image
print(image.shape)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.3, minSize=(30, 30))
for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

eye_detections = eye_detector.detectMultiScale(image_gray, scaleFactor = 1.075, minNeighbors=8, maxSize=(50, 50))
for (x, y, w, h) in eye_detections:
  print(w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2_imshow(image)

# detecting faces with the help of HOG alg

import dlib

image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/people2.jpg')
cv2_imshow(image)

face_detector_HOG = dlib.get_frontal_face_detector() # no need to import file unlike cascade

detections = face_detector_HOG(image, 1) # no need to grayscaling the image unlike cascade
# the second parameter in this func is similar to scaleFactor, the higher it is the smaller the binding boxes

detections # empty array of rectangles[] = no detection

len(detections)

for face in detections:
  #print(face)
  print(face.left())
  print(face.right())
  print(face.top())
  print(face.bottom())

for face in detections:
  l, r, t, b = face.left(), face.right(), face.top(), face.bottom()
  cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)

cv2_imshow(image)

# detecting faces with CNN

!pip install dlib --upgrade --force-reinstall

image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/people2.jpg')

cnn_detector = dlib.cnn_face_detection_model_v1('/content/drive/MyDrive/Computer Vision/Weights/mmod_human_face_detector.dat')

detections = cnn_detector(image, 1)
for face in detections:
  l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
  print(c)
  # the larger the confidence is, the better the detection is
  cv2.rectangle(image, (l, t), (r, b), (255, 0, 0), 2)

cv2_imshow(image)



