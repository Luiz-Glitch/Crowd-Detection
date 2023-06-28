import cv2
import numpy as np
import time
import torch
from utils import points
import pandas as pd
import datetime as dt
import mysql.connector
from sqlalchemy import create_engine
import os

'''
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='l'
                       pretrained=True, verbose=False)
# cuda (:0)
'''
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
model = torch.hub.load('yolo', 'yolov5s', source='local',
                       pretrained=True, verbose=False)
model.cuda('cuda:0')

filename = r'videos\vid_21-12_intervalo2.avi'
# IF: "http://root:cam@pf.373@192.168.150.38/mjpg/video.mjpg?..."
# PC: r'C:/Users/lbess/Desktop/SDD in 3D (copy)/output/'
imgDB_path = r'output\21-12 10fps int2/'
# IF: 'C:/Users/user/Desktop/SDD in 3D/imageDB/'
database = r'C:\Users\lbess\Desktop\SDD in 3D (copy)\database.csv'

def crop_crowd(groups, croppingGrps, coordinates):
    # coordinates[x1, y1, x2, y2]   
    #           min / min / max / max
    print("\nInside crop_crowd: ")
    print("Groups: ", groups)

    grpCoords = []
    for group in croppingGrps:
        # get all the coordinates of all people in the group
        allCoords = np.array([coordinates[person] for person in groups[group][1]])
        # find the min() values of the first 2 rows
        coords1 = [min(i) for i in zip(*allCoords[:, :2])]
        # find the max() values of the last 2 rows
        coords2 = [max(i) for i in zip(*allCoords[:, 2:])]
        # add the group coords to all cropping groups coords
        grpCoords.append(coords1 + coords2)

    return grpCoords

def group_people(X, Y):
    agglomerations = [[0]]
    if len(X) > 0 or len(Y) > 0:
        print("Pairs agglomerated: ")
        print(X, Y)
        for i, j in zip(X, Y):
            for k in agglomerations:
                for l in k:
                    if i in k:
                        if j not in k:
                            k.append(j)
                            break
                    elif j in k:
                        if i not in k:
                            k.append(i)
                            break
                    else:
                        for k in agglomerations:
                            if i in k or j in k:
                                break
                        else:
                            agglomerations.append([i])
                            break

    return agglomerations

def group_tracking(groups, centroids, crowd_cache):
    """ Tracks groups by frames returning new or increased groups.

    ID assignment: for to each group, assigns it the ID of the closest
    one of the previous frame. In case it runs out of past groups to
    compare with, create a new ID.

    New groups: if a new ID is created or, one of the past groups has 
    grown in size, the respective group is added to crop_groups list."""

    # Calculate the average coordinates (centroids) of each group
    grpCentroids = []
    for group in groups: # groups = [[0, 2, 3], [1, 4]] [[0]]
        # Consider only the coordinates of the people in the group
        peopleCoords = [centroids[person] for person in group]
        # Calculate the mean of all coordinates in that group
        centroid = np.mean(peopleCoords, axis=0)
        # Append them to the list with the coordinates of all groups
        grpCentroids.append(centroid)

    # Assign the ID's of the last group coordinates or create new ones
    '''Now, we need to loop through the groups, and, for each one,
    assign it the ID of nearest past centroid. In case there's no other
    centroid to compare with, we need to create a new ID.'''
    for i in range(len(grpCentroids)):
        stopped = False
        # Check for past group centroids to compare with.
        if i < len(crowd_cache):
            '''Here, we don't start the distance measureament from the beggining
            of crowd_cache[], but by the next position from where we've 
            last added a group centroid of the current frame.'''
            clst = i
            for j in range(i+1, len(crowd_cache)):
                eclDist = np.linalg.norm(grpCentroids[i] - crowd_cache[j]['centroid'])
                srtst_eclDist = np.linalg.norm(grpCentroids[i] - crowd_cache[clst]['centroid'])
                clst = j if eclDist < srtst_eclDist else clst

            # If the distance between the centroids is 0, it means it has stopped,
            # probably because the group has ceased to move. Therefore, we can mark 
            # them not to update them.
            dist = np.linalg.norm(grpCentroids[i] - crowd_cache[clst]['centroid'])
            if dist == 0:
                stopped = True

            # ID assignment
            identity = crowd_cache[clst]['id']

            # Delete the closest past centroid to update the list with the ith group
            # afterwards.
            del crowd_cache[clst]
        else:
            '''If there are no past centroids to compare with, that means we have
            new groups and we must create new ID's for them. Also, we need to pass
            that information to crop_groups so that we can do the cropping.'''
            try:
                mycursor.execute("SELECT MAX( crowd_id ) FROM crowd_records;")
                id = mycursor.fetchone()[0]
            except:
                id = -1
            identity = id + 1
            print(50*"=", f"\nNew group created: {i}!\n", 50*"=")

        
        # Pass the ID to the groups to do cropping with new groups later on
        groups[i] = {'id' : identity,
                     'people' : groups[i]}

        # Update crowd_cache for the next group tracking, except if
        # the group has stopped.
        if stopped:
            pass
        else:
            group = {'id' : identity,
                     'centroid' : grpCentroids[i],
                     'size' : len(groups[i]['people'])}
            crowd_cache.insert(i, group)

    
def calculate_distance(point1, point2):
    '''Calculate usual distance.'''
    x1, y1 = point1
    x2, y2 = point2
    return np.linalg.norm([x1 - x2, y1 - y2])

def convert_to_bird(centers, M):
    '''Apply the perpective to the bird's-eye view.'''
    centers = [cv2.perspectiveTransform(np.float32([[center]]), M) for center in centers.copy()]
    centers = [list(center[0, 0]) for center in centers.copy()]
    return centers

def bird_detect_people_on_frame(img, confidence, distance, width, height, database, prevCenters,
                                region=None, dst=None):
    results = model([img[:, :, ::-1]])  # Pass the frame through the model and get the boxes

    xyxy = results.xyxy[0].cpu().numpy()  # xyxy are the box coordinates
    #          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
    # tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
    #         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
    #         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
    #         [9.81310e+02, 3.10712e+02, 1.03111e+03, 4.19273e+02, 2.86850e-01, 2.70000e+01]])

    xyxy = xyxy[xyxy[:, 4] >= confidence]  # Filter desired confidence
    xyxy = xyxy[xyxy[:, 5] == 0]  # Consider only people
    xyxy = xyxy[:, :4]

    # Calculate the centers of the circles
    # They will be the centers of the bottom of the boxes
    centers = []
    for x1, y1, x2, y2 in xyxy:
        center = [np.mean([x1, x2]), y2]
        centers.append(center)

    # We create two transformations
    if region is None:
        # The region on the original image
        region = np.float32([[144, 130], [222, 129], [width, height], [0, height]])
    else:
        region = np.float32(region)
    if dst is None:
        # The rectangle we want the image to be trasnformed to
        dst = np.float32([[0, 0], [width, 0], [width, 3*width], [0, 3*width]])
    else:
        dst = np.float32(dst)
    # The first transformation is straightforward: the region to the rectangle
    # as thin the example before
    M = cv2.getPerspectiveTransform(region, dst)

    # The second transformation is a trick, because, using the common transformation,
    # we can't draw circles at left of the region.
    # This way, we flip all things and draw the circle at right of the region,
    # because we can do it.
    region_flip = region*np.float32([-1, 1]) + np.float32([width, 0])
    dst_flip = dst*np.float32([-1, 1]) + np.float32([width, 0])
    M_flip = cv2.getPerspectiveTransform(region_flip, dst_flip)

    # Convert to bird
    # Now, the center of the circles will be positioned on the rectangle
    # and we can calculate the usual distance
    bird_centers = convert_to_bird(centers, M)

    # We verify if the circles colide
    # If so, they will be red
    colors = ['green']*len(bird_centers)
    X = []
    Y = []
    print('XYXY coords: ')
    for i in range(len(bird_centers)):
        print(f'{i}: ', xyxy[i])
        x, y, w, h = xyxy[i]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(img, (x,y), (w, h), (0, 0, 255), 2)
        for j in range(i+1, len(bird_centers)):
            dist = calculate_distance(bird_centers[i], bird_centers[j])
            if dist < distance:
                colors[i] = 'red'
                X.append(i)
                colors[j] = 'red'
                Y.append(j)
    
    groups = group_people(X, Y)

    if len(groups) > 0:
        print("Groups organized: ", groups)
    
    
    print('Previous centers (1): ', crowd_cache)

    # We draw the circles
    # Because we have two transformation, we will start with two empty
    # images ("overlay" images) to draw the circles
    overlay = np.zeros((3*width, 4*width, 3), np.uint8)
    overlay_flip = np.zeros((3*width, 4*width, 3), np.uint8)
    for i, bird_center in enumerate(bird_centers):
        if colors[i] == 'green':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        x, y = bird_center
        x = int(x)
        y = int(y)
        if x >= int(distance/2+15/2):
            # If it's the case the circle is inside or at right of our region
            # we can use the normal overlay image
            overlay = cv2.circle(overlay, (x, y), int(distance/2),
                                  color, 15, lineType=cv2.LINE_AA)
        else:
            # If the circle is at left of the region,
            # we draw the circle inverted on the other overlay image
            x = width - x
            overlay_flip = cv2.circle(overlay_flip, (x, y), int(distance/2),
                                  color, 15, lineType=cv2.LINE_AA)

    # We apply the inverse transformation to the overlay
    overlay = cv2.warpPerspective(overlay, M, (width, height),
                                  cv2.INTER_NEAREST, cv2.WARP_INVERSE_MAP)
    cv2.imwrite("overlays\overlay.png", overlay)

    # We apply the inverse of the other transformation to the other overlay
    overlay_flip = cv2.warpPerspective(overlay_flip, M_flip, (width, height),
                                       cv2.INTER_NEAREST, cv2.WARP_INVERSE_MAP)
    # Now we "unflip" what the second overlay
    overlay_flip = cv2.flip(overlay_flip, 1)
    cv2.imwrite("overlays\overlay_flip.png", overlay_flip)

    # We add all images
    img = cv2.addWeighted(img, 1, overlay, 1, 0)
    img = cv2.addWeighted(img, 1, overlay_flip, 1, 0)

    # Find new or altered groups to crop
    if len(groups[0]) > 1:
        group_tracking(groups, bird_centers, crowd_cache)
        print('bird_centers: ', bird_centers)
        print('crowd_cache: ', crowd_cache)

        time = dt.datetime.now()

        sizes = []
        ids = []
        for group in groups:
            ids.append(group['id'])
            sizes.append(len(group['people']))

        # print("Crowds detected!")
        # print("Sizes: ", sizes)
        imgName = time.strftime('%y%m%d_%H%M%S')
        cv2.imwrite(imgDB_path+imgName+'.jpg', img)

        d = {
            'crowd_id': ids,
            'size' : sizes,
            'rec_time': time,
            'image': imgName
        }
        data = pd.DataFrame(data=d)
        print(data)
        connection = create_engine('mysql+pymysql://root:lu_iz@localhost:3306/crowd_detection')
        data.to_sql(con=connection, name="crowd_records", if_exists='append', index=False)
        data.to_csv(database, mode='a', header=False)
        data.to_excel('dados.xls', index = False)

    else:
        crowd_cache.clear()
    
    return img

output = cv2.VideoWriter('output.avi', 
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        5, (1200, 720))


crowd_cache = []

db = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="sql",
                database="crowd_detection"
            )
            
mycursor = db.cursor()
mycursor.execute('SET autocommit = ON;')

def bird_detect_people_on_video(filename, database, confidence=0.9, distance=160):
    # Capture video
    cap = cv2.VideoCapture(filename)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Comment this line and the import of points if you want to use a 
    # predefined region
    region = points.capture_points()

    counter = 0
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        frame= cv2.resize(frame,(1200, 720),interpolation=cv2.INTER_CUBIC)
        width, height = (1200, 720)
        
        print()
        if cv2.waitKey(30) & 0xFF == ord('.'):
            break

        if ret == True:
            counter += 1
            if counter == 10:
                # Detect people as a bird
                frame = bird_detect_people_on_frame(frame, confidence, distance,
                                                    width, height, database, crowd_cache, region)
                counter = 0
            # Write frame to new video
            output.write(frame)

            # Display frame
            cv2.imshow("cam-mov",frame)
        else:
            break
            
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

bird_detect_people_on_video(filename, database, confidence=0.5, distance=1800)