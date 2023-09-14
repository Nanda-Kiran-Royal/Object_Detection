import cv2 as cv


# img = cv.imread('lena.jpeg')
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
classNames = []
classFile = 'coco.names'
#
# with open(classFile,'rt') as f:
#     classNames = f.read()#.rstrip('\n')#.strip('\n')
#     #classNames = list(classNames)
#     print(classNames)


# Open the "coco.names" file and read its contents
with open("coco.names", "rt") as file:
    classNames = [line.strip() for line in file]

# Strip whitespace and create a single-line list



# Print the single-line list to verify
print((classNames))



configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

weightsPath = 'frozen_inference_graph.pb'

net = cv.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5 )
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds,confs,bbox = net.detect(img,confThreshold = 0.5)
    print(classIds,bbox)


    for classId,confidence,box in zip(classIds,confs,bbox):
        cv.rectangle(img,box,(0,255,0),3)

        cv.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                   cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)



    cv.imshow('Out',img)
    cv.waitKey(1)