from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import picar

#Hello world

picar.setup()
# Show image captured by camera, True to turn on, youwill need #DISPLAY and it also slows the speed of tracking
show_image_enable   = True
draw_circle_enable  = True
scan_enable         = True
rear_wheels_enable  = True
front_wheels_enable = True
pan_tilt_enable     = True

kernel = np.ones((5,5),np.uint8)
img = cv2.VideoCapture(-1)

SCREEN_WIDTH = 700
SCREEN_HIGHT = 700
img.set(3,SCREEN_WIDTH)
img.set(4,SCREEN_HIGHT)
CENTER_X = SCREEN_WIDTH/2
CENTER_Y = SCREEN_HIGHT/2
BALL_SIZE_MIN = SCREEN_HIGHT/15
BALL_SIZE_MAX = SCREEN_HIGHT/2

# Filter setting, DONOT CHANGE
hmn = 12
hmx = 37
smn = 96
smx = 255
vmn = 186
vmx = 255

# camera follow mode:
# 0 = step by step(slow, stable), 
# 1 = calculate the step(fast, unstable)
follow_mode = 0

CAMERA_STEP = 20
CAMERA_X_ANGLE = 20
CAMERA_Y_ANGLE = 20

MIDDLE_TOLERANT = 5
PAN_ANGLE_MAX   = 170
PAN_ANGLE_MIN   = 10
TILT_ANGLE_MAX  = 150
TILT_ANGLE_MIN  = 70
FW_ANGLE_MAX    = 90+30
FW_ANGLE_MIN    = 90-30

SCAN_POS = [[20, TILT_ANGLE_MIN], [50, TILT_ANGLE_MIN], [90, TILT_ANGLE_MIN], [130, TILT_ANGLE_MIN], [160, TILT_ANGLE_MIN], 
            [160, 80], [130, 80], [90, 80], [50, 80], [20, 80]]

bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)
picar.setup()

fw.offset = 0
pan_servo.offset = 10
tilt_servo.offset = 0

bw.speed = 0
fw.turn(90)
pan_servo.write(90)
tilt_servo.write(90)

motor_speed = 20

def nothing(x):
    pass

# driving
def main():
    count = 0
    total = 100
    delta = 0
    av_angle = 0
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('test.avi',fourcc, 25, (640,480), True)
    # cv2.VideoWriter(['testvideo', cv2.CV_FOURCC('M','J','P','G'), 25, 
    #           (640,480),True])
    
    try:
        
        while count < total:
            print(count)
            a, orig_image = img.read()
            print("Channels", orig_image.shape)
            
            
            frame = orig_image
            
            # take R out of BGR
            b,g,r = cv2.split(frame)
            
            # gaussian blur
            blurred = cv2.GaussianBlur(r,(5,5),0)
            
            # canny
            Threshold1 = 15;
            Threshold2 = 200;
            FilterSize = 1
            E = cv2.Canny(blurred, Threshold1, Threshold2, FilterSize)

            # lines
            # Rres = 1
            # Thetares = 1*np.pi/180
            # Threshold = 100
            # minLineLength = 10
            # maxLineGap = 100
            # cv2.HoughLines(edges, 1, np.pi / 180, 190)
            # (edges, lines, 1, CV_PI / 180, 100, 100, 10);
            # lines = cv2.HoughLinesP(E,3, np.pi/180, 100, 100, 10)
            max_slider = 100
            lines = cv2.HoughLinesP(E, 1, np.pi/180, max_slider, minLineLength=50, maxLineGap=50)

            max_x = 640
            max_y = 480
            
            if lines is not None:
                N = lines.shape[0]
                serious_lines = []
                
                left_xs = []
                right_xs = []

                for i in range(N):
                    x1, y1, x2, y2 = lines[i][0]

                    dx = x2-x1
                    dy = y2-y1
                    angle = np.arctan2(dy,dx) * (180/np.pi)
                    length = np.sqrt(dx**2+dy**2)

                    delta = 60
                    if abs(angle) > 90-delta and abs(angle) < 90+delta:
                        color = (0,0,255)
                        
                        dx = abs(x2)-abs(x1)
                        dy = abs(y2)-abs(y1)
                        dxdy = abs(dx/dy)

                        y3 = 0 
                        
                        if angle < 0:
                            # left line
                            color = (0, 255, 0)
                            x3 = x2 + int(dxdy*y2)
                            left_xs.append(x3)
                        else:
                            # right line
                            x3 = x2 - int(dxdy*y2)
                            right_xs.append(x3)
                                
                        cv2.line(frame,(x2,y2),(x3,y3),(255, 0, 0),2)
                        
                        cv2.line(frame,(x1,y1),(x2,y2),color,2)
                        
                        print("Original end point", x2, y2)
                        print("Start point", x1, y1)
                        print("Angle", angle)
                        print("Length", length)

                        serious_lines.append((x1, y1, angle, length))

                sorted_lines = sorted(serious_lines, key=lambda tup: tup[2])
            #     print(sorted_lines)

                actual_lines = []
                
                position = (10,430)
                
                def avg(val):
                    if len(val) > 0:
                        return np.average(val)
                
                def write(txt, lineno=1):
                    cv2.putText(frame, txt, (10,360+40*lineno), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                
                x_left = avg(left_xs)
                x_right = avg(right_xs)
                
                if x_left is not None and x_right is not None:
                    space_left = x_left
                    space_right = 640-x_right
                    space_diff = space_left-space_right
                    write(f'Diff {space_diff} ({space_left}, {space_right})')  
                    
                    if abs(space_diff) > 35:
                        if space_diff > 0:
                            fw.turn(110)
                            write('Steer right', 2)
                        else:
                            fw.turn(70)
                            write('Steer left', 2)
                    else:
                        fw.turn(90)
                elif x_left is not None:
                    space_left = int(x_left)
                    space_diff = space_left-300
                    write(f'Space left = {space_left} / diff {space_diff}')
                    
                    if abs(space_diff) > 35:
                        if space_diff > 0:
                            fw.turn(105)
                            write('Steer right', 2)
                        else:
                            fw.turn(75)
                            write('Steer left', 2)
                    else:
                        fw.turn(90)
                elif x_right is not None:
                    space_right = int(640-x_right)
                    space_diff = 340-space_right
                    write(f'Space right = {space_right} / diff {space_diff}')
                    
                    if abs(space_diff) > 35:
                        if space_diff > 0:
                            fw.turn(105)
                            write('Steer right', 2)
                        else:
                            fw.turn(75)
                            write('Steer left', 2)
                    else:
                        fw.turn(90)
                else:
                    write('No data')
                    
                    
                

                prev = 0
                for x1,y2,angle,length in sorted_lines:
                    delta = np.abs(np.abs(prev)-np.abs(angle))
                    if delta > 1:
                        actual_lines.append([[x1, x2, angle, length]])
                    else:
                        actual_lines[-1].append([x1, x2, angle, length])
                    prev = angle

                print("Final lines", actual_lines)
                #print("Final lines", list(zip(actual_lines[1])))
            else:
                print("No lines")
            
            #print(a)
            #print(orig_image)
            
            out.write(orig_image)
            cv2.imwrite(f"./frames/{count}.jpg", orig_image)
            cv2.namedWindow("View3", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("View3", 640, 480)
            cv2.imshow("View3", orig_image)
        
            
            k = cv2.waitKey(5) & 0xFF
            
            #if count < (total / 2):
            print("Forward")
               # fw.turn(130)
            bw.speed = 35
            bw.forward()
            #else:
            #    print("Backward")
            #    fw.turn(60)
            #    bw.speed = min(100, 20 + (count-(total/2)) * 1)
            #    bw.backward()
            count += 1
            sleep(0.04)
        
    finally:
        out.release()
        fw.turn(90)
        bw.speed = 0
        cv2.waitKey(0)                 # Waits forever for user to press any key
        cv2.destroyAllWindows()

def main_park_turns():
    count = 0
    total = 100
    
    try:
        while count < total:
            print(count)
            a, orig_image = img.read()
            print(a)
            #print(orig_image)
            cv2.namedWindow("View3", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("View3", 700, 700)
            cv2.imshow("View3", orig_image)
            
            
            cv2.imwrite(f"./frames/{count}.jpg", orig_image)
            
            k = cv2.waitKey(5) & 0xFF
            
            if count < (total / 2):
                print("Forward")
                fw.turn(130)
                bw.speed = min(100, 20 + count * 1)
                bw.forward()
            else:
                print("Backward")
                fw.turn(60)
                bw.speed = min(100, 20 + (count-(total/2)) * 1)
                bw.backward()
            count += 1
            sleep(0.04)
    finally:
        fw.turn(90)
        bw.speed = 0

def main2():
    pan_angle = 90              # initial angle for pan
    tilt_angle = 90             # initial angle for tilt
    fw_angle = 90

    scan_count = 0
    print("Begin!")
    while True:
        x = 0             # x initial in the middle
        y = 0             # y initial in the middle
        r = 0             # ball radius initial to 0(no balls if r < ball_size)

        for i in range(10):
            (tmp_x, tmp_y), tmp_r = find_blob()
            if tmp_r > BALL_SIZE_MIN:
                x = tmp_x
                y = tmp_y
                r = tmp_r
                break

        print(x, y, r)

        # scan:
        if r < BALL_SIZE_MIN:
            bw.stop()
            if scan_enable:
                #bw.stop()
                pan_angle = SCAN_POS[scan_count][0]
                tilt_angle = SCAN_POS[scan_count][1]
                if pan_tilt_enable:
                    pan_servo.write(pan_angle)
                    tilt_servo.write(tilt_angle)
                scan_count += 1
                if scan_count >= len(SCAN_POS):
                    scan_count = 0
            else:
                sleep(0.1)
            
        elif r < BALL_SIZE_MAX:
            if follow_mode == 0:
                if abs(x - CENTER_X) > MIDDLE_TOLERANT:
                    if x < CENTER_X:                              # Ball is on left
                        pan_angle += CAMERA_STEP
                        #print("Left   ", )
                        if pan_angle > PAN_ANGLE_MAX:
                            pan_angle = PAN_ANGLE_MAX
                    else:                                         # Ball is on right
                        pan_angle -= CAMERA_STEP
                        #print("Right  ",)
                        if pan_angle < PAN_ANGLE_MIN:
                            pan_angle = PAN_ANGLE_MIN
                if abs(y - CENTER_Y) > MIDDLE_TOLERANT:
                    if y < CENTER_Y :                             # Ball is on top
                        tilt_angle += CAMERA_STEP
                        #print("Top    " )
                        if tilt_angle > TILT_ANGLE_MAX:
                            tilt_angle = TILT_ANGLE_MAX
                    else:                                         # Ball is on bottom
                        tilt_angle -= CAMERA_STEP
                        #print("Bottom ")
                        if tilt_angle < TILT_ANGLE_MIN:
                            tilt_angle = TILT_ANGLE_MIN
            else:
                delta_x = CENTER_X - x
                delta_y = CENTER_Y - y
                #print("x = %s, delta_x = %s" % (x, delta_x))
                #print("y = %s, delta_y = %s" % (y, delta_y))
                delta_pan = int(float(CAMERA_X_ANGLE) / SCREEN_WIDTH * delta_x)
                #print("delta_pan = %s" % delta_pan)
                pan_angle += delta_pan
                delta_tilt = int(float(CAMERA_Y_ANGLE) / SCREEN_HIGHT * delta_y)
                #print("delta_tilt = %s" % delta_tilt)
                tilt_angle += delta_tilt

                if pan_angle > PAN_ANGLE_MAX:
                    pan_angle = PAN_ANGLE_MAX
                elif pan_angle < PAN_ANGLE_MIN:
                    pan_angle = PAN_ANGLE_MIN
                if tilt_angle > TILT_ANGLE_MAX:
                    tilt_angle = TILT_ANGLE_MAX
                elif tilt_angle < TILT_ANGLE_MIN:
                    tilt_angle = TILT_ANGLE_MIN
            
            if pan_tilt_enable:
                pan_servo.write(pan_angle)
                tilt_servo.write(tilt_angle)
            sleep(0.01)
            fw_angle = 180 - pan_angle
            if fw_angle < FW_ANGLE_MIN or fw_angle > FW_ANGLE_MAX:
                fw_angle = ((180 - fw_angle) - 90)/2 + 90
                if front_wheels_enable:
                    fw.turn(fw_angle)
                if rear_wheels_enable:
                    bw.speed = motor_speed
                    bw.backward()
            else:
                if front_wheels_enable:
                    fw.turn(fw_angle)
                if rear_wheels_enable:
                    bw.speed = motor_speed
                    bw.forward()
        else:
            bw.stop()
            img.release()
        
def destroy():
    bw.stop()
    img.release()

def test():
    fw.turn(90)

def find_blob() :
    radius = 0
    # Load input image
    _, bgr_image = img.read()

    orig_image = bgr_image

    bgr_image = cv2.medianBlur(bgr_image, 3)

    # Convert input image to HSV
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image, keep only the red pixels
    lower_red_hue_range = cv2.inRange(hsv_image, (80, 100, 100), (100, 255, 255))
    upper_red_hue_range = cv2.inRange(hsv_image, (140, 100, 100), (179, 255, 255))
    # Combine the above two images
    red_hue_image = cv2.addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0)

    red_hue_image = cv2.GaussianBlur(red_hue_image, (9, 9), 2, 2)

    # Use the Hough transform to detect circles in the combined threshold image
    circles = cv2.HoughCircles(red_hue_image, cv2.HOUGH_GRADIENT, 1, 120, 100, 20, 10, 0);

    # Loop over all detected circles and outline them on the original image
    all_r = np.array([])
    if circles is not None:
        for i in circles[0]:

            all_r = np.append(all_r, int(round(i[2])))
        closest_ball = all_r.argmax()
        center=(int(round(circles[0][closest_ball][0])), int(round(circles[0][closest_ball][1])))
        radius=int(round(circles[0][closest_ball][2]))
        if draw_circle_enable:
            cv2.circle(orig_image, center, radius, (255, 0, 0), 5);

    # Show images
    if show_image_enable:
        cv2.namedWindow("Threshold lower image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Threshold lower image", lower_red_hue_range)
        cv2.namedWindow("Threshold upper image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Threshold upper image", upper_red_hue_range)
        cv2.namedWindow("Combined threshold images", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Combined threshold images", red_hue_image)
        cv2.namedWindow("Detected red circles on the input image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Detected red circles on the input image", orig_image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        return (0, 0), 0
    if radius > 3:
        return center, radius
    else:
        return (0, 0), 0


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        destroy()
