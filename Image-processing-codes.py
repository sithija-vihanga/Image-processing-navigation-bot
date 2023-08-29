import cv2
import urllib.request
import numpy as np
import socket
import math

from collections import OrderedDict

# Define a color name lookup table
color_names = OrderedDict({
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    #"cyan": (0, 255, 255),
    #"magenta": (255, 0, 255),
    "white": (255, 255, 255),
    #"black": (0, 0, 0),
    # Add more colors as needed
})


# Replace the URL with the IP camera's stream URL
Ip_addr = '192.168.43.86'
url = 'http://' + Ip_addr + '/640x480.jpg'
cv2.namedWindow("Live Cam Testing", cv2.WINDOW_AUTOSIZE)

# IP address and port of the ESP32-CAM server
ESP32_CAM_IP = Ip_addr
ESP32_CAM_PORT = 12345

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if the IP camera stream is opened successfully
if not cap.isOpened():
    print("Failed to open the IP camera stream")
    exit()

def get_closest_color_name(rgb_color):
    # Calculate the Euclidean distance between the RGB color and each color in the lookup table
    distances = {color: np.linalg.norm(np.array(rgb_color) - np.array(color_value)) for color, color_value in color_names.items()}

    # Get the closest color name
    closest_color = min(distances, key=distances.get)
    return closest_color

def detect_contour_colors(image_path):
    # Read the image
    image = frame_with_contours

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on their area and shape
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # For each contour, compute the average color within the contour's region
    for contour in filtered_contours:
        # Create a mask for the current contour
        mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Compute the mean color within the contour's region
        mean_color = cv2.mean(image, mask=mask)[:3]  # Slice the mean_color to get B, G, R channels

        # Convert the mean_color from BGR to RGB format
        mean_color_rgb = tuple(reversed(mean_color))

        # Get the closest color name
        closest_color_name = get_closest_color_name(mean_color_rgb)

        # Draw the bounding box with the detected color label
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Color: {closest_color_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Show the result
    cv2.imshow("Contour Colors Detection", image)
  

# Threshold trackbar callback function
def on_threshold_change(val):
    global threshold_value
    threshold_value = val

# Gaussian blur trackbar callback function
def on_blur_change(val):
    global blur_value
    blur_value = val if val % 2 != 0 else val + 1  # Ensure that the blur value is odd

# Minimum contour area threshold (initial value)
min_contour_area = 100

# Function to update the minimum contour area
def on_min_contour_area_change(val):
    global min_contour_area
    min_contour_area = val


def detect_arrow_direction(image_path):
    # Load the image
    image = threshold_resized
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if ((cv2.contourArea(contour) >100) and (cv2.contourArea(contour) <20000))]
    
    for contour in filtered_contours:
        # Approximate the contour with a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has 7 points (arbitrary number for an arrow shape)
        if len(approx) == 7:
            # Compute the centroid of the arrow
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get the first and last points of the contour
            first_point = approx[0][0]
            last_point = approx[-1][0]
            
            # Calculate the direction vector
            direction_vector = (last_point - first_point)
            
            # Calculate the angle with the horizontal axis
            angle_rad = np.arctan2(direction_vector[1], direction_vector[0])
            angle_deg = np.degrees(angle_rad)
            
            # Print the arrow direction
            if -40 <= angle_deg < 5:
                #print("Arrow is pointing Left")
                cv2.putText(image, 'Downward arrow ', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif 50 <= angle_deg < 140:
                #print("Arrow is pointing Up")
                cv2.putText(image, 'Right Arrow ', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            elif -130 <= angle_deg < -40:
                #print("Arrow is pointing Down")
                cv2.putText(image, 'Left Arrow ', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            else:
                #print("Arrow is pointing Right")
                cv2.putText(image, 'Upward Arrow ', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw the contour and arrow direction on the original image
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
            cv2.putText(image, f"{angle_deg:.2f} deg", (cx - 50, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show the result
    cv2.imshow("Arrow Detection", image)
 

    

# Create a separate window for the trackbars
cv2.namedWindow('Threshold Trackbar')
cv2.namedWindow('Gaussian Blur Trackbar')
cv2.namedWindow('Min Contour Area Trackbar')
cv2.createTrackbar('Threshold', 'Threshold Trackbar', 127, 255, on_threshold_change)
cv2.createTrackbar('Gaussian Blur', 'Gaussian Blur Trackbar', 5, 20, on_blur_change)
cv2.createTrackbar('Min Contour Area', 'Min Contour Area Trackbar', min_contour_area, 1000, on_min_contour_area_change)

msgAvailable = False

while True:
    # Read a frame from the video stream
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the Gaussian blur value from the trackbar
    blur_value = cv2.getTrackbarPos('Gaussian Blur', 'Gaussian Blur Trackbar')

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)

    # Get the threshold value from the trackbar
    threshold_value = cv2.getTrackbarPos('Threshold', 'Threshold Trackbar')

    # Apply thresholding to the blurred image
    _, threshold = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on their area (greater than the threshold)
    filtered_contours = [contour for contour in contours if ((cv2.contourArea(contour) > min_contour_area) and (cv2.contourArea(contour) <20000))]

    # Draw bounding boxes around the filtered contours on the original frame
    frame_with_contours = frame.copy()
    radar_view = np.zeros((250, 400, 3), dtype=np.uint8)

    radar_width, radar_height = 400, 250

    radar_center_x = radar_width // 2
    radar_center_y = radar_height - 1

    cv2.circle(radar_view, (radar_center_x, radar_center_y), 10, (0, 255, 0), 1)
    cv2.circle(radar_view, (radar_center_x, radar_center_y), 40, (0, 255, 0), 1)
    cv2.circle(radar_view, (radar_center_x, radar_center_y), 80, (0, 255, 0), 1)
    cv2.circle(radar_view, (radar_center_x, radar_center_y), 120, (0, 255, 0), 1)
    cv2.circle(radar_view, (radar_center_x, radar_center_y), 160, (0, 255, 0), 1)
    cv2.circle(radar_view, (radar_center_x, radar_center_y), 200, (0, 255, 0), 1)

    cv2.drawContours(frame_with_contours, filtered_contours, -1, (0, 255, 0), 2)
    for contour in filtered_contours:
        
        x, y, w, h = cv2.boundingRect(contour)
        #cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw small circles at the center of the detected contours in red
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(radar_view, (int(cx / 640 * 320), int(cy / 480 * 240)), 3, (0, 0, 255), -1)

        # Draw red dots on the radar view for detected contours
        angle = math.atan2((cx / 640 * 320) - 160, 240 - (cy / 480 * 240))
        radius = 3 #int(math.sqrt(((cx / 640 * 320) - 160) * 2 + (240 - (cy / 480 * 240)) * 2))

    # Draw a rectangle on the original frame
    height, width = frame.shape[:2]
    middle_x = width // 2
    bottom_y = height
    rect_x = middle_x - 180  # Half of the rectangle length
    rect_y = bottom_y - 100  # Distance from the bottom (adjust as needed)
    rect_width = 256
    rect_height = 6
    #cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 0, 0), -1)

    # Resize the images for cascading
    frame_resized = cv2.resize(frame, (320, 240))
    gray_resized = cv2.resize(gray, (320, 240))
    blurred_resized = cv2.resize(blurred, (320, 240))
    threshold_resized = cv2.resize(threshold, (320, 240))

    # Convert grayscale and thresholded images to 3 channels
    gray_resized = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    threshold_resized = cv2.cvtColor(threshold_resized, cv2.COLOR_GRAY2BGR)

    # Convert all images to the same number of channels
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    gray_resized = cv2.cvtColor(gray_resized, cv2.COLOR_BGR2RGB)
    blurred_resized = cv2.cvtColor(blurred_resized, cv2.COLOR_BGR2RGB)
    threshold_resized = cv2.cvtColor(threshold_resized, cv2.COLOR_BGR2RGB)
    detect_arrow_direction(threshold_resized)
    detect_contour_colors(frame_with_contours)

    # Add the image names in blue color
    cv2.putText(frame_resized, 'Original Frame', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(gray_resized, 'Grayscale', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(blurred_resized, 'Blurred', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(threshold_resized, 'Thresholded', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Create larger windows for contour and radar views
    contour_view = cv2.resize(frame_with_contours, (640, 480))
    radar_large = cv2.resize(radar_view, (640, 480))

    # Create a vertical stack of images
    top_row = np.hstack((frame_resized, gray_resized, blurred_resized, threshold_resized))
    bottom_row = np.hstack((contour_view, radar_large))
    output = np.vstack((top_row, bottom_row))
    output = cv2.resize(output, (1350, 800))
    # Show the cascaded images and the trackbar windows
    cv2.imshow('Cascaded Images', output)
    cv2.imshow('Threshold Trackbar', np.zeros((1, 400), np.uint8))  # Empty black window for the threshold trackbar
    cv2.imshow('Gaussian Blur Trackbar', np.zeros((1, 400), np.uint8))  # Empty black window for the blur trackbar
    cv2.imshow('Min Contour Area Trackbar', np.zeros((1, 400), np.uint8))  # Empty black window for the min contour area trackbar
    
    
    # ... (Rest of the code remains unchanged)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('w'):         #Forward
        msgAvailable = True
        message = 'F'
    elif key == ord('a'):         #left
        msgAvailable = True
        message = 'L'
    elif key == ord('d'):         #right
        msgAvailable = True
        message = 'R'
    elif key == ord('s'):         #reverse
        msgAvailable = True
        message = 'B'
    if msgAvailable:
        msgAvailable = False
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Connect to the server (ESP32-CAM)
            client_socket.connect((ESP32_CAM_IP, ESP32_CAM_PORT))

            # Send the message
            client_socket.sendall(message.encode())

        finally:
            # Close the socket
            client_socket.close()

cap.release()
cv2.destroyAllWindows()
