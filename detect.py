import cv2

# Load the pre-trained cascade classifier for pedestrian detection
cascade = cv2.CascadeClassifier('pedestrian.xml')

# Define the tracker algorithm
tracker = cv2.TrackerKCF_create()

# Start the video capture from the default camera
cap = cv2.VideoCapture(0)

# Read the first frame from the video
ret, frame = cap.read()

# Initialize the bounding box for tracking
bbox = None

while True:
    # Read the current frame from the video
    ret, frame = cap.read()
    
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if bbox is None:
            # Detect pedestrians in the current frame using the cascade classifier
            pedestrians = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            if len(pedestrians) > 0:
                # Select the first detected pedestrian as the object to track
                x, y, w, h = pedestrians[0]
                bbox = (x, y, w, h)
                
                # Initialize the tracker with the current bounding box and image
                tracker.init(frame, bbox)
        
        else:
            # Track the object in the next frame and update the bounding box
            success, bbox = tracker.update(frame)
            
            if success:
                # Draw a bounding box around the tracked object
                x, y, w, h = [int(coord) for coord in bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            else:
                # Put the text on the screen saying “LOST”
                cv2.putText(frame, "LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                bbox = None
                
        # Display the video feed with bounding box and tracking status
        cv2.imshow('Pedestrian Tracking', frame)
        
    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
