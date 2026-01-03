import cv2
from ultralytics import YOLO

# 1. Load Model (Standard COCO)
# Class 0 is 'person' - highly accurate, no training needed
print("Loading Safety System...")
model = YOLO('yolov8n.pt')

# 2. Setup Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 3. Define the "Danger Zone" (ROI)
# Let's make the top-left corner the "Machine Area"
# Format: (x_min, y_min, x_max, y_max)
ZONE_COORDS = (50, 50, 300, 300) 

def is_overlapping(box1, box2):
    """
    Simple collision detection.
    Returns True if the Person box overlaps the Danger Zone.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # If one rectangle is to the left of the other
    if x1_max < x2_min or x2_max < x1_min:
        return False
    # If one rectangle is above the other
    if y1_max < y2_min or y2_max < y1_min:
        return False
        
    return True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run Inference
    results = model(frame, stream=True, verbose=False)
    
    # Default State: SYSTEM SAFE (Green)
    zone_color = (0, 255, 0) 
    safety_status = "RUNNING"
    trigger_stop = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # We ONLY check for Class 0 (Person)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == 0 and conf > 0.5:
                # Get Person Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw box around person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

                # Check for Collision with Danger Zone
                person_box = (x1, y1, x2, y2)
                if is_overlapping(person_box, ZONE_COORDS):
                    trigger_stop = True
                    zone_color = (0, 0, 255) # RED
                    safety_status = "EMERGENCY STOP"
                    
                    # ----------------------------------------------------
                    # [PLACEHOLDER] MODBUS WRITE HERE
                    # client.write_coil(address=1, value=0) # Cut Power
                    # ----------------------------------------------------

    # Draw the Danger Zone
    cv2.rectangle(frame, (ZONE_COORDS[0], ZONE_COORDS[1]), (ZONE_COORDS[2], ZONE_COORDS[3]), zone_color, 3)
    cv2.putText(frame, "DANGER ZONE (MOTOR)", (ZONE_COORDS[0], ZONE_COORDS[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

    # Status Overlay
    cv2.putText(frame, f"STATUS: {safety_status}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 3)

    cv2.imshow('Rockwell Safety Curtain Demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()