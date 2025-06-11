#!/usr/bin/env python3
import cv2
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from collections import deque
import logging

# Define Class IDs
COCO_PERSON_CLASS = 0          # COCO
COCO_MOTORCYCLE_CLASS = 3      # COCO
CUSTOM_HELMET_CLASS = 4        # Custom model
CUSTOM_SCOOTER_CLASS = 6       # Custom model

# For displaying hazardous elements (Class IDs from the Custom model)
DISPLAY_CLASSES = {
    0: "weapon",
    1: "garbage_bag",
    2: "Fire",
    3: "Smoke"
}

# All class names
CLASS_NAMES = {
    0: "weapon",
    1: "garbage_bag",
    2: "Fire",
    3: "Smoke",
    4: "Helmet",
    5: "motorcycle",  # COCO
    6: "electric scooter"
}

# Color settings (BGR)
COLORS = {
    "default": (180, 0, 255),
    "person": (0, 255, 0),
    "helmet_ok": (0, 255, 0),
    "no_helmet": (0, 0, 255),
    "helmet": (255, 255, 0),
    "motorcycle": (255, 200, 0),
    "scooter": (0, 255, 255)
}

# Other constants
OFFSET_Y = 50                # Amount to extend the vehicle bounding box upward
FRAME_THRESHOLD = 10         # Trigger when each condition persists for 5 frames or more
IOU_THRESHOLD = 0.3          # IoU comparison threshold

logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Function to compute IoU
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

class YOLODetector(Node):
    def __init__(self):
        super().__init__("yolo")
        # Video file path (can be passed as a parameter)
        self.declare_parameter("video_path", "/ros2_workspace/neubility_test_dataset/fight-20250402T045503Z-001/fight/31-1_cam01_fight03_place02_night_summer.mp4")
        video_path = self.get_parameter("video_path").value

        self.get_logger().info("Loading YOLO models...")
        # Load the COCO model and the custom model (merged.pt)
        self.coco_model = YOLO("yolo11n.pt")
        self.custom_model = YOLO("merged.pt")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error("Error opening video")
            rclpy.shutdown()

        self.resize_w, self.resize_h = 640, 360
        self.bridge = CvBridge()
        # Publishers: one for the visualized image and one for the image triggering VLM inference
        self.image_pub = self.create_publisher(Image, "detection/image", 10)
        self.vlm_image_pub = self.create_publisher(Image, "input_image", 10)
        
        # Timer: operates at approximately 30 FPS
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.frame_idx = 0

        # Queues for each condition (store frame images)
        self.queue_riders = deque()     # queue_riders: store frames where vehicle riding count (motorcycle/scooter) >= 2
        self.queue_nohelmet = deque()   # queue_nohelmet: store frames with at least one person not wearing a helmet
        # Create a queue for each hazardous element (weapon, garbage_bag, Fire, Smoke); key is label, value is a deque
        self.queue_danger = {label: deque() for label in DISPLAY_CLASSES.values()}

    def trigger_vlm_inference(self, frame, condition: str):
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.vlm_image_pub.publish(image_msg)
        self.get_logger().info(f"Triggered VLM inference ({condition}) by publishing input_image.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("No more frames available, shutting down...")
            self.cap.release()
            rclpy.shutdown()
            return

        self.frame_idx += 1
        frame = cv2.resize(frame, (self.resize_w, self.resize_h))

        # Run model inference
        coco_results = self.coco_model(frame)[0]
        custom_results = self.custom_model(frame)[0]

        # Extract people and motorcycles from COCO results (using COCO_PERSON_CLASS and COCO_MOTORCYCLE_CLASS)
        people = [b for b in coco_results.boxes.data if int(b[5]) == COCO_PERSON_CLASS]
        motorcycles = [b for b in coco_results.boxes.data if int(b[5]) == COCO_MOTORCYCLE_CLASS]

        # Separate helmet, scooter, and others (e.g., hazardous elements) from custom results
        custom_boxes = custom_results.boxes.data
        helmets = [b for b in custom_boxes if int(b[5]) == CUSTOM_HELMET_CLASS]
        scooters = [b for b in custom_boxes if int(b[5]) == CUSTOM_SCOOTER_CLASS]
        others = [b for b in custom_boxes if int(b[5]) not in [CUSTOM_HELMET_CLASS, CUSTOM_SCOOTER_CLASS]]

        # Handling hazardous elements: if an object corresponding to a DISPLAY_CLASSES label is detected,
        # save a copy of the current frame to the respective queue.
        for b in custom_boxes:
            cls_id = int(b[5])
            if cls_id in DISPLAY_CLASSES:
                label = DISPLAY_CLASSES[cls_id]
                # Save a copy of the current frame
                self.queue_danger[label].append(frame.copy())
                self.get_logger().info(f"Queue_{label} length: {len(self.queue_danger[label])}")

        # Vehicles: combine motorcycles detected from COCO and scooters from the custom model
        vehicles = motorcycles + scooters
        riding_count = 0
        nohelmet_flag = False
        
        # For each vehicle, detect passengers (people) by expanding the vehicle bounding box and checking IoU with people boxes
        for v in vehicles:
            vx1, vy1, vx2, vy2 = map(int, v[:4])
            expanded_box = (vx1, max(0, vy1 - OFFSET_Y), vx2, vy2)
            passengers = []
            for p in people:
                px1, py1, px2, py2 = map(int, p[:4])
                person_box = (px1, py1, px2, py2)
                if iou(expanded_box, person_box) > IOU_THRESHOLD:
                    passengers.append(person_box)
            # If passengers are detected within the vehicle's vicinity, increase riding_count
            if passengers:
                riding_count += len(passengers)
                # Check among the passengers if anyone is not wearing a helmet
                for person_box in passengers:
                    has_helmet = False
                    for h in helmets:
                        hx1, hy1, hx2, hy2 = map(int, h[:4])
                        if iou(person_box, (hx1, hy1, hx2, hy2)) > IOU_THRESHOLD:
                            has_helmet = True
                            break
                    if not has_helmet:
                        nohelmet_flag = True

            # Visualize the vehicle bounding box and number of passengers
            cv2.rectangle(frame, (expanded_box[0], expanded_box[1]), (expanded_box[2], expanded_box[3]), (255, 255, 255), 2)
            cv2.putText(frame, f"Passengers: {len(passengers)}", (vx1, max(0, vy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Condition 1: If vehicle has at least 2 passengers, store the frame in queue_riders
        if riding_count >= 2:
            self.queue_riders.append(frame.copy())
            self.get_logger().info(f"queue_riders length: {len(self.queue_riders)}")
        # Condition 2: If any passenger is not wearing a helmet, store the frame in queue_nohelmet
        if nohelmet_flag:
            self.queue_nohelmet.append(frame.copy())
            self.get_logger().info(f"queue_nohelmet length: {len(self.queue_nohelmet)}")
        
        # If the length of each queue reaches FRAME_THRESHOLD, call trigger_vlm_inference() and clear the queue
        if len(self.queue_riders) >= FRAME_THRESHOLD:
            self.trigger_vlm_inference(frame, condition="vehicle riders >=2")
            self.queue_riders.clear()
        if len(self.queue_nohelmet) >= FRAME_THRESHOLD:
            self.trigger_vlm_inference(frame, condition="no helmet detected")
            self.queue_nohelmet.clear()
        # Process each hazardous element queue
        for label, queue_obj in self.queue_danger.items():
            if len(queue_obj) >= FRAME_THRESHOLD:
                self.trigger_vlm_inference(frame, condition=f"{label} detected")
                queue_obj.clear()

        # Visualize user-defined classes (excluding Helmet and Scooter)
        for b in others:
            cls_id = int(b[5])
            x1, y1, x2, y2 = map(int, b[:4])
            label = CLASS_NAMES.get(cls_id, f"id:{cls_id}")
            color = COLORS.get(label.lower(), COLORS["default"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Visualize people
        for p in people:
            px1, py1, px2, py2 = map(int, p[:4])
            cv2.rectangle(frame, (px1, py1), (px2, py2), COLORS["person"], 2)
            cv2.putText(frame, "Person", (px1, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["person"], 2)

        # Hazardous element text display (shown in the top-right corner)
        detected_labels = {DISPLAY_CLASSES[int(b[5])] for b in custom_boxes if int(b[5]) in DISPLAY_CLASSES}
        for i, label in enumerate(sorted(detected_labels)):
            text = f"Detected: {label}"
            cv2.putText(frame, text, (self.resize_w - 200, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display statistics in the top-left corner
        cv2.putText(frame, f"Total People: {len(people)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["person"], 2)
        cv2.putText(frame, f"Riding People: {riding_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Publish and display the final result image (for visualization)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))
        cv2.imshow("Final Detection Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    node.get_logger().info("Waiting for 30 seconds before starting YOLO detection...")
    time.sleep(30)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO node...")
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
