import cv2
import numpy as np

url = "http://192.168.1.12:81/stream"
KNOWN_DIAMETER = 7.0  
FOCAL_LENGTH = 500    

def calculate_distance(diameter_pixels):
    if diameter_pixels == 0:
        return 0
    return (KNOWN_DIAMETER * FOCAL_LENGTH) / diameter_pixels


cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("error can't reach to stream")
    exit()

print("start streaming successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("faildFrame")
        break

   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)

    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        if radius > 10:  
            diameter_pixels = radius * 2
            distance = calculate_distance(diameter_pixels)

            
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
           
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            
            text = f"Distance: {distance:.2f} cm"
            cv2.putText(frame, text, (int(x) - 50, int(y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    
    cv2.imshow("Ball Detection & Depth Estimation", frame)
    cv2.imshow("Mask (Color Detection)", mask)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()