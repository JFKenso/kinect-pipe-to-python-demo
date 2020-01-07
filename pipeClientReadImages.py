
#  python .\pipeClientReadImages.py -m .\haarcascade_frontalface_default.xml -n .\mobilenet_ssd\MobileNetSSD_deploy.caffemodel -p .\mobilenet_ssd\MobileNetSSD_deploy.prototxt

# NamedPipe
import win32file

import numpy as np
import os, os.path

# For visualization
import cv2
import matplotlib.pyplot as plt


import asyncio, io, glob, sys, time, uuid, requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType

import argparse

from datetime import datetime
import urllib, time
import urllib.request as urllib2
import grequests
import requests_async as requests
import concurrent.futures
from requests_futures.sessions import FuturesSession

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.4,	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skipframes", type=int, default=30,	help="the number frames to skip")
ap.add_argument("-m", "--model", required=True,	help = "path to where the face cascade resides")    
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-n", "--netmodel", required=True, help="path to Caffe pre-trained model")
args = vars(ap.parse_args())

# The image size of depth/ir
# Assuming depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED, change it otherwise
FRAME_WIDTH = 640
FRAME_HEIGHT = 576
BYTES_PER_PIXEL = 2

# For gray visulization
MAX_DEPTH_FOR_VIS = 8000.0
MAX_AB_FOR_VIS = 512.0

#Azure Face 
KEY = "200a79d197984c4cbe4d57157fbf4be8"
ENDPOINT = "https://australiaeast.api.cognitive.microsoft.com/"
PERSON_GROUP_ID = 'melb_partners2'
SKIP_FRAMES = args['skipframes']

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

detector = cv2.CascadeClassifier(args["model"])

#MobileNet_SSD classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["netmodel"])
peopleCounterAvg = 0
peopleCounterList = []
happinessScoreAvg = 0
happinessScoreList = []
ageAvg = 0
ageList = []
maleCounter = 0
femaleCounter = 0
unknownCounter = 0
FRAMES_BACK = 2

#PowerBI Reporting
session = FuturesSession()

KINECT_PEOPLE_COUNT_URL = "https://api.powerbi.com/beta/d22a5dd1-ae22-4d29-8b30-bdb2c0f25ea4/datasets/ca6f544e-7336-489b-8776-fd8c5843f587/rows?key=dRFfF8BYCmG%2Bm2d%2BvH8qNjitW794JPSECZmbmeL2ly5OBO98csqFehdBGVb1aeT5SAhE%2BEusCDyCNFKb85p0dQ%3D%3D"
    # Example:
    #curl --include --request POST --header "Content-Type: application/json" --data-binary "[{\"peopleCounter\" :98.6,\"peopleCounterAvg\" :98.6,\"timestamp\" :\"2019-10-31T00:50:40.677Z\"}]" "https://api.powerbi.com/beta/d22a5dd1-ae22-4d29-8b30-bdb2c0f25ea4/datasets/ca6f544e-7336-489b-8776-fd8c5843f587/rows?key=dRFfF8BYCmG%2Bm2d%2BvH8qNjitW794JPSECZmbmeL2ly5OBO98csqFehdBGVb1aeT5SAhE%2BEusCDyCNFKb85p0dQ%3D%3D"

KINECT_PEOPLE_INFO_URL = "https://api.powerbi.com/beta/d22a5dd1-ae22-4d29-8b30-bdb2c0f25ea4/datasets/433aca4c-0d3c-476b-b774-6ce11077834b/rows?key=hSoBIfw%2FkiE4%2FlQ4bEyT1GIBSmmNpB7KPmiQksG06WJzTYXA3KKggYDgQVab6jQ%2FStyCJaBTSJPt1AR3JC%2Bnzw%3D%3D"
    # Example:
    #curl --include \
    #--request POST \
    #--header "Content-Type: application/json" \
    #--data-binary "[
    #{
    #\"age\" :98.6,
    #\"gender\" :\"AAAAA555555\",
    #\"emotion\" :\"AAAAA555555\",
    #\"timestamp\" :\"2019-10-31T00:57:16.442Z\"
    #}
    #]" \
    #"https://api.powerbi.com/beta/d22a5dd1-ae22-4d29-8b30-bdb2c0f25ea4/datasets/433aca4c-0d3c-476b-b774-6ce11077834b/rows?key=hSoBIfw%2FkiE4%2FlQ4bEyT1GIBSmmNpB7KPmiQksG06WJzTYXA3KKggYDgQVab6jQ%2FStyCJaBTSJPt1AR3JC%2Bnzw%3D%3D"

# Convert width height to a point in a rectangle
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    bottom = left + rect.height
    right = top + rect.width
    return (left, top, bottom, right)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def proc_response(response, **kwargs):
    # do something ..
    print ("========Processing response=============" + str(response.request.url))
    print (str(response))
    if response.status_code != 200:
        print (str(response.request.url))
        print (str(response.content))

def exception_handler(request, exception):
    print("Exception Found: "+ str(exception))

def load_url3(url, params, headers, timeout):
    r = session.post(url, data=payload, headers={'Content-Type':'application/json'})
    #response_one = r.result()
    #print('response one status: {0}'.format(response_one.status_code))
    #print(response_one.content)
    return r

def load_url(url, params, headers, timeout):
    r = requests.post(url, data=payload, headers={'Content-Type':'application/json'}, timeout = timeout)
    print("Request Response: "+ str(r))
    return r

def load_url2(url, params, headers, timeout):
    print("Received request with params: "+ str(params))
    urls = [url] 
    rs = (grequests.post(u, data=payload, headers={'Content-Type':'application/json'}) for u in urls)
    requests = grequests.map(rs)
    for response in requests:
        print(response)

def get_urls():
    return ["url1","url2"]

def postRequest(url, params, headers):
    print("Calling URL: "+ url)



    try: 
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        #    future_to_url = executor.submit(load_url, url, params, headers, 10)
        #    future = concurrent.futures.as_completed(future_to_url)
        
        #    print(future)
        
        
        
            executor.submit(load_url3, url, params, headers, 10)
        #req = grequests.post(KINECT_PEOPLE_COUNT_URL, data=params, headers={'Content-Type':'application/json'})
        #job = grequests.send(req, grequests.Pool(1))
        #urls = [url]
        
        #rs = (grequests.post(u, data=payload, headers={'Content-Type':'application/json'}) for u in urls)
        #requests = grequests.map(rs)
        #for response in requests:
        #    print(response)
    except urllib2.HTTPError as e:
        print("HTTP Error: {0} - {1}".format(e.code, e.reason))
    except urllib2.URLError as e:
        print("URL Error: {0}".format(e.reason))
    #except Exception as e:
        #print("General Exception: {0}".format(e))

if __name__ == "__main__":

    # Create pipe client
    fileHandle = win32file.CreateFile("\\\\.\\pipe\\mynamedpipe",
        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
        0, None,
        win32file.OPEN_EXISTING,
        0, None)

    # For visualization
    #cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('vis', FRAME_WIDTH, FRAME_HEIGHT)

    frame_counter = 0

    displayText = ''

    while True:
        try:
            # Send request to pipe server
            request_msg = "Request depth image and ir image"
            win32file.WriteFile(fileHandle, request_msg.encode())
            # Read reply data, need to be in same order/size as how you write them in the pipe server in pipe_streaming_example/main.cpp
            depth_data = win32file.ReadFile(fileHandle, FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL)
            ab_data = win32file.ReadFile(fileHandle, FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL)
            # Reshape for image visualization
            depth_img_full = np.frombuffer(depth_data[1], dtype=np.uint16).reshape(FRAME_HEIGHT, FRAME_WIDTH).copy()
            ab_img_full = np.frombuffer(ab_data[1], dtype=np.uint16).reshape(FRAME_HEIGHT, FRAME_WIDTH).copy()
            
            depth_vis = (plt.get_cmap("gray")(depth_img_full / MAX_DEPTH_FOR_VIS)[..., :3]*255.0).astype(np.uint8)
            ab_vis = (plt.get_cmap("gray")(ab_img_full / MAX_AB_FOR_VIS)[..., :3]*255.0).astype(np.uint8)
            
            #cv2.imshow("ab_img_full", ab_img_full)
            #cv2.imshow("depth_img_full", depth_img_full)

            # Visualize the images
            #vis = np.hstack([depth_vis, ab_vis])
            ab_vis = cv2.flip( ab_vis, 1 )
            vis = np.hstack([ab_vis])
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            frame_counter += 1

            img = vis.copy()
            #img = np.array(test) 

            # buf will be the encoded image
            ret,buf = cv2.imencode('.jpg', img)

            # stream-ify the buffer
            stream = io.BytesIO(buf)
            
            if frame_counter % SKIP_FRAMES == 0:
                # Mobilenet_SSD for people counting
                blob = cv2.dnn.blobFromImage(vis, 0.007843, (FRAME_WIDTH, FRAME_HEIGHT), 127.5)
                net.setInput(blob)
                detections = net.forward()
                
                peopleCounter = 0

                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence = detections[0, 0, i, 2]

                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    #print ("Detection Found: " + str(CLASSES[idx]) + " confidence: " + str(confidence))

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # filter out weak detections by requiring a minimum
                    # confidence
                    if confidence > args["confidence"]:
                        peopleCounter = peopleCounter + 1
                
                peopleCounterList.append(peopleCounter)
                if(len(peopleCounterList) > FRAMES_BACK):
                    peopleCounterList.pop(0)
                
                peopleCounterAvg = moving_average(peopleCounterList, FRAMES_BACK)
                if peopleCounterAvg.size == 0:
                    peopleCounterAvg = 0
                else:
                    peopleCounterAvg = peopleCounterAvg[0]

                # Send payload to PowerBI for reporting
                now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%Z")

                urls = [KINECT_PEOPLE_COUNT_URL]

                payload = '[{"peopleCounter" :' + str(peopleCounter) + ',"peopleCounterAvg" :' + str(peopleCounterAvg) + ',"timestamp" :"' + str(now) + 'Z"}]'
                #payload = "[{\"peopleCounter\" :98.6,\"peopleCounterAvg\" :98.6,\"timestamp\" :\"2019-10-31T00:50:40.677Z\"}]"
                print(payload)
                postRequest(KINECT_PEOPLE_COUNT_URL, payload, "'Content-Type':'application/json'")
                
                
                #displayText = 'There are {} people detected!'.format(peopleCounter)
                #print(displayText)

                #displayText = 'There rolling average is {} !'.format(peopleCounterAvg)
                #print(displayText)

                #HAR to detect faces
                rects = detector.detectMultiScale(vis, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                #print(rects)
                #print(len(rects))
                num_faces = len(rects)

                if (num_faces > 0):   
                    # Detect faces
                    face_ids = []
                    faces = face_client.face.detect_with_stream(stream, return_face_id=True, return_face_attributes=['age','gender','emotion','facialHair','glasses','exposure','makeup'])

                    

                    for face in faces:
                        face_ids.append(face.face_id)
                        age = face.face_attributes.age
                        gender = str(face.face_attributes.gender)
                        happinessScore = face.face_attributes.emotion.happiness

                        if(happinessScore == 0):
                            happinessScore = 0.5

                        # Calculate Average Rolling Happiness
                        happinessScoreList.append(happinessScore)
                        if(len(happinessScoreList) > FRAMES_BACK):
                            happinessScoreList.pop(0)
                
                        happinessScoreAvg = moving_average(happinessScoreList, FRAMES_BACK)
                        if happinessScoreAvg.size == 0:
                            happinessScoreAvg = 0
                        else:
                            happinessScoreAvg = happinessScoreAvg[0]

                        if(len(happinessScoreList) < FRAMES_BACK):
                            happinessScoreAvg = 0.5

                        # Calculate Average Rolling Age
                        ageList.append(age)
                        if(len(ageList) > FRAMES_BACK):
                            ageList.pop(0)
                
                        ageAvg = moving_average(ageList, FRAMES_BACK)
                        if ageAvg.size == 0:
                            ageAvg = 0
                        else:
                            ageAvg = ageAvg[0]

                        # Rolling Sum of Women and Men
                        if("Gender.male" in gender):
                            maleCounter += 1
                        elif("Gender.female" in gender):
                            femaleCounter += 1
                        else:
                            unknownCounter += 1
                        
                        payload = '[{"age" :' + str(ageAvg) + ',"female" :' + str(femaleCounter) + ',"male" :' + str(maleCounter) + ',"happinessScore" :' + str(happinessScoreAvg) + ',"timestamp" :"' + str(now) + 'Z"}]'
                        print(payload)
                        postRequest(KINECT_PEOPLE_INFO_URL, payload, "'Content-Type':'application/json'")
                        
                        print('FACE ATTRIBUTES: ' + str(face.face_attributes))
                        print('FACE HAPPINESS: {} happiness probability={}'.format(face.face_id, face.face_attributes.emotion.happiness))

                    if False:
                        # Identify faces
                        results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
                        print('Identifying faces')
                        for person in results:
                            print("Person Results: "+ str(person))
                            if person.candidates:
                                cand = person.candidates[0]
                                print(cand)
                        if not results:
                            print('No person identified in the person group.')
                    #else:
                        #print('No faces found')

            #else:
                #print("Skipping frame %d".format(frame_counter))
            #cv2.putText(vis, displayText, (FRAME_HEIGHT, FRAME_WIDTH - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            #cv2.imshow("vis", vis)
            #cv2.imshow("stream", stream)

            key = cv2.waitKey(1)
            if key == 27: # Esc key to stop
                break 
        except Exception as e:
            print("General Exception: {0}".format(e))


    win32file.CloseHandle(fileHandle)


