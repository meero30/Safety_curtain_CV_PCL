import cv2
import time
from ultralytics import YOLO

# 1. Load Model
print("Loading Safety System...")
model = YOLO('yolov8n.pt')

# 2. Setup Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    if x1_max < x2_min or x2_max < x1_min: return False
    if y1_max < y2_min or y2_max < y1_min: return False
    return True

# --- HELPER: DRAW INSTRUCTIONS ---

def draw_setup_instructions(frame):
    cv2.rectangle(frame, (0, 0), (640, 70), (0, 0, 0), -1)
    
    cv2.putText(frame, "SETUP MODE", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Instructions Column
    cv2.putText(frame, "Press 'S' to Draw Zone", (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'SPACE' to Confirm", (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'C' to Cancel / 'Q' to Quit", (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_run_dashboard(frame, fps, status, is_danger):
    h, w, _ = frame.shape
    # Top Bar
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # Status Color
    color = (0, 0, 255) if is_danger else (0, 255, 0)
    
    # Left: Status
    cv2.putText(frame, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Right: FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Bottom: Instructions
    cv2.putText(frame, "PRESS 'Q' TO QUIT", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# --- INTERACTIVE ZONE SELECTION ---
ZONE_COORDS = None

while True:
    ret, frame = cap.read()
    if ret:
        draw_setup_instructions(frame)
        
        cv2.imshow('Rockwell Safety Curtain Setup', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            cv2.destroyWindow('Rockwell Safety Curtain Setup')
            
            roi = cv2.selectROI("SETUP: Draw Danger Zone", frame, fromCenter=False, showCrosshair=True)
            
            x_start, y_start, width, height = roi
            ZONE_COORDS = (x_start, y_start, x_start + width, y_start + height)
            
            # Check for cancellation
            if roi[2] == 0 or roi[3] == 0:
                cv2.destroyWindow("SETUP: Draw Danger Zone")
                print("Selection Cancelled. Press 's' to try again.")
                continue 

            cv2.destroyWindow("SETUP: Draw Danger Zone")
            print(f"Zone Confirmed: {ZONE_COORDS}")
            print("Starting AI...")
            break

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
    else:
        print("Error: Could not read frame.")
        exit()


# --- AI RUN LOOP ---
prev_time = 0

while True:
    curr_time = time.time()
    ret, frame = cap.read()
    if not ret: break

    # FPS Calculation
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # Run Inference
    results = model(frame, stream=True, verbose=False)
    
    zone_color = (0, 255, 0) 
    safety_status = "SAFE"
    trigger_stop = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Check Collision
                if is_overlapping((x1, y1, x2, y2), ZONE_COORDS):
                    trigger_stop = True
                    zone_color = (0, 0, 255) # RED
                    safety_status = "E-STOP"
                
                # Draw Person (Red if dangerous, Orange if safe)
                p_color = (0, 0, 255) if trigger_stop else (255, 100, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), p_color, 2)

    # Draw Danger Zone
    cv2.rectangle(frame, (ZONE_COORDS[0], ZONE_COORDS[1]), (ZONE_COORDS[2], ZONE_COORDS[3]), zone_color, 3)
    cv2.putText(frame, "DANGER ZONE", (ZONE_COORDS[0], ZONE_COORDS[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

    # ADDED: Draw Dashboard
    draw_run_dashboard(frame, fps, safety_status, trigger_stop)

    cv2.imshow('Rockwell Safety Curtain (Running)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()