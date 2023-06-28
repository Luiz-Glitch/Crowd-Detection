import cv2 as cv

# create our coordinates lists
coordinates = []

# draw lines
def draw_lines(frame):
    for i in range(len(coordinates) - 1): 
        frame = cv.line(frame, coordinates[i], coordinates[i+1], (0, 0, 255), 2)
    if len(coordinates) == 4:
        frame = cv.line(frame, coordinates[3], coordinates[0], (0, 0, 255), 2)

# get coordinates of the clicked point
def select_points(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        point = [x, y]
        print("Points: [{}, {}]".format(x, y))
        if len(coordinates) < 4:
            print("Points added!")
            coordinates.append(point)

# prepare window where the mouse events monitoring will take place
cv.namedWindow("Video", cv.WINDOW_AUTOSIZE)
cv.setMouseCallback("Video", select_points)

def capture_points(filename, w_size):
    # set video capture to webcam
    cap = cv.VideoCapture(filename)

    while cap.isOpened(): # check if the camera is opened
        while True:
            ret, frame = cap.read()
            frame= cv.resize(frame, w_size,interpolation=cv.INTER_CUBIC)

            # check if the pixel was correctly read
            if ret == False:
                break
            
            draw_lines(frame)
            cv.imshow("Video", frame)

            if len(coordinates) == 4:
                break

            key = cv.waitKey(1) & 0xff
            if key == ord('.'):
                break

        cap.release()
        cv.destroyAllWindows()
    else:
        print("Couldn't open camera")
        
    return coordinates