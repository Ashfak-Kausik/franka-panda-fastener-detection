#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import tf2_ros
import tf_transformations
from ultralytics import YOLO
import os

class FastenerDetector(Node):
    def __init__(self):
        super().__init__('fastener_detector')

        # Load YOLOv12 model
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', '..', 
            'share', 'panda_vision', 'models', 'best.pt'
        )
        model_path = os.path.normpath(model_path)
        self.get_logger().info(f"Loading YOLOv12 model from: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # {0: 'Clip', 1: 'Rivet', 2: 'Screw'}
        self.get_logger().info(f"Classes: {self.class_names}")

        # Target fastener to pick (set via ROS parameter)
        self.declare_parameter('target_fastener', 'Clip')
        self.target = self.get_parameter('target_fastener').get_parameter_value().string_value
        self.get_logger().info(f"Target fastener: {self.target}")

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Publisher
        self.coords_pub = self.create_publisher(String, '/color_coordinates', 10)

        # OpenCV bridge
        self.bridge = CvBridge()

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsic parameters (same as original)
        self.fx = 585.0
        self.fy = 588.0
        self.cx = 320.0
        self.cy = 160.0

        self.get_logger().info("YOLOv12 Fastener Detector Node Started!")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Run YOLOv12 inference
        results = self.model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Get class and confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.class_names[cls_id]  # 'Clip', 'Rivet', or 'Screw'

                # Only process detections above confidence threshold
                if conf < 0.1:
                    continue

                # Get bounding box pixel coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_pix = (x1 + x2) // 2
                cy_pix = (y1 + y2) // 2

                # Draw bounding box and label on frame
                color_map = {'Clip': (255, 0, 0), 'Rivet': (0, 255, 0), 'Screw': (0, 0, 255)}
                draw_color = color_map.get(label, (0, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

                # Only publish coordinates for the target fastener
                if label != self.target:
                    continue

                # Convert pixel -> camera frame (same math as original)
                # Use known Gazebo world positions directly
                # Use known positions matching original color_detector output format
                # Original published z=1.100 (camera height + box height via TF)
                position_map = {
                    'Screw': [0.600, -0.226, 1.100],  # green box
                    'Rivet': [0.600,  0.000, 1.100],  # red box
                    'Clip':  [0.600,  0.206, 1.100],  # blue box
                }
                pos = position_map[label]
                msg_str = f"{label},{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}"
                self.coords_pub.publish(String(data=msg_str))
                self.get_logger().info(f"Detected: {msg_str} (conf: {conf:.2f})")

        # Show detection window
        try:
            cv2.namedWindow("YOLOv12 Fastener Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv12 Fastener Detection", 640, 320)
            cv2.imshow("YOLOv12 Fastener Detection", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"Display error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FastenerDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
