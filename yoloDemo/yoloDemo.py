import cv2
import numpy as np

cap = cv2.VideoCapture(1)
whT = 320
confThreshold = 0.5  # 50% confidenceThreshold
nmsThreshold = 0.2

# LOAD MODEL
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  # looping through the coco dataset label for every object
# print(classNames)

# Model Files
modelConfiguration = "yolov3.cfg"  # the configuration for the detectable objects
modelWeights = "yolov3.weights"  # the weights for the detectable objects
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    heightT, widthT, centerT = img.shape
    boundBox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * widthT), int(det[3] * heightT)
                x, y = int((det[0] * widthT) - w / 2), int((det[1] * heightT) - h / 2)
                boundBox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundBox, confs, confThreshold, nmsThreshold)

    # drawing a bounding box around the object
    for i in indices:
        i = i[0]
        box = boundBox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    print(layersNames)
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
