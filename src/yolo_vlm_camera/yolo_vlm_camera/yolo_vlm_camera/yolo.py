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

# Class IDs and constants definition
COCO_PERSON_CLASS = 0          # COCO model person class ID
COCO_MOTORCYCLE_CLASS = 3      # COCO model motorcycle class ID
CUSTOM_HELMET_CLASS = 4        # Custom model helmet class ID
CUSTOM_SCOOTER_CLASS = 6       # Custom model scooter class ID

# Display mapping for dangerous objects (using custom model class IDs)
DISPLAY_CLASSES = {
    0: "weapon",
    1: "garbage_bag",
    2: "Fire",
    3: "Smoke"
}

# Full class names for visualization
CLASS_NAMES = {
    0: "weapon",
    1: "garbage_bag",
    2: "Fire",
    3: "Smoke",
    4: "Helmet",
    5: "motorcycle",  # from COCO
    6: "electric scooter"
}

# Color definitions (BGR format)
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
OFFSET_Y = 50                # Amount to expand vehicle bounding box upward
FRAME_THRESHOLD = 10         # Number of frames to accumulate before triggering VLM
IOU_THRESHOLD = 0.3          # IoU threshold to consider a person inside a vehicle

logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Function to calculate Intersection over Union (IoU) between two boxes
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
        # Use camera input: set the default video_path parameter to "/dev/video0"
        self.declare_parameter("video_path", "/dev/video0")
        video_path = self.get_parameter("video_path").value

        self.get_logger().info("Loading YOLO models...")
        # Load the COCO model and the custom merged model
        self.coco_model = YOLO("yolo11n.pt")
        self.custom_model = YOLO("merged.pt")

        # Open camera stream from /dev/video0
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error("Error opening camera at " + video_path)
            rclpy.shutdown()

        # (Optional) Uncomment to set camera resolution manually
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        self.resize_w, self.resize_h = 640, 360
        self.bridge = CvBridge()
        # Publisher for visualized detection images
        self.image_pub = self.create_publisher(Image, "detection/image", 10)
        # Publisher for VLM inference: frames meeting trigger conditions are sent here
        self.vlm_image_pub = self.create_publisher(Image, "input_image", 10)
        
        # Create a timer to call timer_callback at approximately 30 FPS
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.frame_idx = 0

        # Queues to hold frames for each condition
        self.queue_riders = deque()     # Frames with vehicle riders >= 2
        self.queue_nohelmet = deque()   # Frames with at least one rider without a helmet
        # Separate queue for each dangerous object (weapon, garbage_bag, Fire, Smoke)
        self.queue_danger = {label: deque() for label in DISPLAY_CLASSES.values()}

    def trigger_vlm_inference(self, frame, condition: str):
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.vlm_image_pub.publish(image_msg)
        self.get_logger().info(f"Triggered VLM inference ({condition}) by publishing input_image.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("No frames received from camera, shutting down...")
            self.cap.release()
            rclpy.shutdown()
            return

        self.frame_idx += 1
        frame = cv2.resize(frame, (self.resize_w, self.resize_h))

        # Run inference with both COCO and custom YOLO models
        coco_results = self.coco_model(frame)[0]
        custom_results = self.custom_model(frame)[0]

        # Extract persons and motorcycles from COCO results
        people = [b for b in coco_results.boxes.data if int(b[5]) == COCO_PERSON_CLASS]
        motorcycles = [b for b in coco_results.boxes.data if int(b[5]) == COCO_MOTORCYCLE_CLASS]

        # From custom model results, extract helmets, scooters, and other objects (dangerous items)
        custom_boxes = custom_results.boxes.data
        helmets = [b for b in custom_boxes if int(b[5]) == CUSTOM_HELMET_CLASS]
        scooters = [b for b in custom_boxes if int(b[5]) == CUSTOM_SCOOTER_CLASS]
        others = [b for b in custom_boxes if int(b[5]) not in [CUSTOM_HELMET_CLASS, CUSTOM_SCOOTER_CLASS]]

        # For each dangerous object (weapon, garbage_bag, Fire, Smoke), store frame copies in respective queues
        for b in custom_boxes:
            cls_id = int(b[5])
            if cls_id in DISPLAY_CLASSES:
                label = DISPLAY_CLASSES[cls_id]
                self.queue_danger[label].append(frame.copy())
                self.get_logger().info(f"Queue_{label} length: {len(self.queue_danger[label])}")

        # Combine motorcycles and scooters as vehicles
        vehicles = motorcycles + scooters
        riding_count = 0
        nohelmet_flag = False
        
        # For each vehicle, detect passengers by expanding the vehicle bounding box upward and calculating IoU with persons
        for v in vehicles:
            vx1, vy1, vx2, vy2 = map(int, v[:4])
            expanded_box = (vx1, max(0, vy1 - OFFSET_Y), vx2, vy2)
            passengers = []
            for p in people:
                px1, py1, px2, py2 = map(int, p[:4])
                person_box = (px1, py1, px2, py2)
                if iou(expanded_box, person_box) > IOU_THRESHOLD:
                    passengers.append(person_box)
            if passengers:
                riding_count += len(passengers)
                # Check if any passenger is missing a helmet
                for person_box in passengers:
                    has_helmet = False
                    for h in helmets:
                        hx1, hy1, hx2, hy2 = map(int, h[:4])
                        if iou(person_box, (hx1, hy1, hx2, hy2)) > IOU_THRESHOLD:
                            has_helmet = True
                            break
                    if not has_helmet:
                        nohelmet_flag = True

            # Draw the expanded vehicle box and display the number of passengers (visualization)
            cv2.rectangle(frame, (expanded_box[0], expanded_box[1]), (expanded_box[2], expanded_box[3]), (255, 255, 255), 2)
            cv2.putText(frame, f"Passengers: {len(passengers)}", (vx1, max(0, vy1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Condition 1: If there are 2 or more riders, add this frame copy to the queue_riders
        if riding_count >= 2:
            self.queue_riders.append(frame.copy())
            self.get_logger().info(f"queue_riders length: {len(self.queue_riders)}")
        # Condition 2: If any rider is without a helmet, add this frame copy to the queue_nohelmet
        if nohelmet_flag:
            self.queue_nohelmet.append(frame.copy())
            self.get_logger().info(f"queue_nohelmet length: {len(self.queue_nohelmet)}")
        
        # If any queue reaches the defined threshold (FRAME_THRESHOLD), trigger VLM inference and clear that queue.
        if len(self.queue_riders) >= FRAME_THRESHOLD:
            self.trigger_vlm_inference(frame, condition="vehicle riders >=2")
            self.queue_riders.clear()
        if len(self.queue_nohelmet) >= FRAME_THRESHOLD:
            self.trigger_vlm_inference(frame, condition="no helmet detected")
            self.queue_nohelmet.clear()
        for label, queue_obj in self.queue_danger.items():
            if len(queue_obj) >= FRAME_THRESHOLD:
                self.trigger_vlm_inference(frame, condition=f"{label} detected")
                queue_obj.clear()

        # Visualization:
        # Draw bounding boxes for other custom detections (excluding helmets and scooters)
        for b in others:
            cls_id = int(b[5])
            x1, y1, x2, y2 = map(int, b[:4])
            label = CLASS_NAMES.get(cls_id, f"id:{cls_id}")
            color = COLORS.get(label.lower(), COLORS["default"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw person bounding boxes
        for p in people:
            px1, py1, px2, py2 = map(int, p[:4])
            cv2.rectangle(frame, (px1, py1), (px2, py2), COLORS["person"], 2)
            cv2.putText(frame, "Person", (px1, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["person"], 2)

        # Display detected dangerous objects labels on the top-right
        detected_labels = {DISPLAY_CLASSES[int(b[5])] for b in custom_boxes if int(b[5]) in DISPLAY_CLASSES}
        for i, label in enumerate(sorted(detected_labels)):
            text = f"Detected: {label}"
            cv2.putText(frame, text, (self.resize_w - 200, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display statistics on the top-left corner
        cv2.putText(frame, f"Total People: {len(people)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["person"], 2)
        cv2.putText(frame, f"Riding People: {riding_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Publish the visualized image and show it on screen
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
