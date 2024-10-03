import cv2

# Load the pre-trained object detection model
net = cv2.dnn.readNetFromCaffe("path/to/deploy.prototxt", "path/to/model.caffemodel")

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Preprocess the frame for the object detection model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Process the detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()