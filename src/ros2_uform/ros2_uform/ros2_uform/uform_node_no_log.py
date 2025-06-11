import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
import torch
import time
from transformers import AutoModel, AutoProcessor

class UformInferenceNode(Node):
    def __init__(self):
        super().__init__('uform_node')
        
        # Publisher for the /output topic to publish inference results
        self.publisher_ = self.create_publisher(String, 'output', 10)
        # Subscriber for the image
        self.subscription = self.create_subscription(
            Image,
            'input_image',
            self.image_callback,
            10)
        self.subscription  

        # Create a CvBridge object for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        self.get_logger().info("Loading model and processor...")
        
        # Check for GPU usage; if on CPU, use torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.benchmark = True
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load the UForm-Gen2 model and processor
        self.model = AutoModel.from_pretrained(
            "unum-cloud/uform-gen2-qwen-500m",
            trust_remote_code=True,
            torch_dtype=dtype
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "unum-cloud/uform-gen2-qwen-500m", 
            trust_remote_code=True
        )

        # Prompt used for inference
        self.prompt = "Analyze the image and determine which category it falls under from the following options: fire, smoke, fallen person, fight, weapon possession, general. Provide only one category as your final answer in English."

        self.get_logger().info("Model and processor loaded successfully.")

    def image_callback(self, msg):
        try:
            self.get_logger().info("Image received. Starting inference...")
            # Convert the sensor_msgs/Image message to an OpenCV image using CvBridge (BGR format)
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge conversion error: {e}")
                return

            # Convert the BGR image to RGB and then convert to a PIL image
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Preprocess the input for the model
            inputs = self.processor(text=[self.prompt], images=[pil_image], return_tensors="pt")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            inputs = {
                "input_ids": inputs["input_ids"].to(self.device, dtype=torch.long),
                "attention_mask": inputs["attention_mask"].to(self.device, dtype=dtype),
                "images": inputs["images"].to(self.device, dtype=dtype)
            }
            for key, value in inputs.items():
                self.get_logger().debug(f"{key}: dtype={value.dtype}, shape={value.shape}")

            # start_time = time.time()

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=20,
                    eos_token_id=151645,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # inference_time = time.time() - start_time

            prompt_len = inputs["input_ids"].shape[1]
            decoded_text = self.processor.batch_decode(output[:, prompt_len:])[0]

            # if self.device == "cuda":
            #     max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            #     max_reserved = torch.cuda.max_memory_reserved() / 1024**2
            #     self.get_logger().info(f"GPU Max Allocated Memory: {max_allocated:.2f} MB")
            #     self.get_logger().info(f"GPU Max Reserved Memory: {max_reserved:.2f} MB")

            # self.get_logger().info(f"Inference Time: {inference_time:.4f} seconds")
            # self.get_logger().info(f"Generated Response: {decoded_text}")

            # Publish the inference result to the /output topic
            output_msg = String()
            output_msg.data = decoded_text
            self.publisher_.publish(output_msg)
            self.get_logger().info("Published output message to /output topic.")
        except Exception as e:
            self.get_logger().error(f"Exception in image callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = UformInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
