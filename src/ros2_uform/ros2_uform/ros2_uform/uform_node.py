import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
import torch
import time
from transformers import AutoModel, AutoProcessor

class UformInferenceNode(Node):
    def __init__(self):
        super().__init__('uform_node')
        # enable or disable logging
        self.declare_parameter('enable_logging', False)
        self.enable_logging = self.get_parameter('enable_logging').get_parameter_value().bool_value
        self.add_on_set_parameters_callback(self._on_param_update)

        # initialize log file
        now = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        self.log_file = f"/ros2_workspace/src/ros2_uform/ros2_uform/logging/log_csv/log_{ts}.csv"
        try:
            with open(self.log_file, 'x') as f:
                f.write("timestamp,duration_s,image_path,anomaly_type\n")
        except FileExistsError:
            pass

        # publisher for inference output
        self.publisher_ = self.create_publisher(String, 'output', 10)
        # subscriber for input images
        self.subscription = self.create_subscription(
            Image,
            'input_image',
            self.image_callback,
            10
        )

        # bridge for converting ROS images
        self.bridge = CvBridge()
        self.start_time = None
        self.keywords = ['fight', 'fire', 'smoke', 'fallen person', 'weapon possession']

        self.get_logger().info("Loading model and processor...")
        # set device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.benchmark = True
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        model_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # load model and processor
        self.model = AutoModel.from_pretrained(
            "unum-cloud/uform-gen2-qwen-500m",
            trust_remote_code=True,
            torch_dtype=model_dtype
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "unum-cloud/uform-gen2-qwen-500m",
            trust_remote_code=True
        )

        # inference prompt
        self.prompt = (
            "Analyze the image and determine which category it falls under from the "
            "following options: fire, smoke, fallen person, fight, weapon possession, general. "
            "Provide only one category as your final answer in English."
        )

        self.get_logger().info("Model and processor loaded successfully.")

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'enable_logging' and p.type_ == Parameter.Type.BOOL:
                self.enable_logging = p.value
                self.get_logger().info(f"enable_logging: {self.enable_logging}")
        return SetParametersResult(successful=True)

    def _detect(self, text):
        t = text.lower()
        return [kw for kw in self.keywords if kw in t]

    def image_callback(self, msg):
        try:
            self.get_logger().info("Image received. Starting inference...")
            # convert ROS image to OpenCV BGR
            try:
                bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge error: {e}")
                return

            # convert BGR to RGB and to PIL image
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)

            # preprocess inputs
            inputs = self.processor(
                text=[self.prompt],
                images=[pil_image],
                return_tensors="pt"
            )
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            inputs = {
                "input_ids": inputs["input_ids"].to(self.device, dtype=torch.long),
                "attention_mask": inputs["attention_mask"].to(self.device, dtype=dtype),
                "images": inputs["images"].to(self.device, dtype=dtype)
            }

            # log input shapes
            for key, value in inputs.items():
                self.get_logger().debug(f"{key}: dtype={value.dtype}, shape={value.shape}")

            # perform inference
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=20,
                    eos_token_id=151645,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

            # decode output
            prompt_len = inputs["input_ids"].shape[1]
            result = self.processor.batch_decode(output[:, prompt_len:])[0].strip()

            # publish result
            msg_out = String()
            msg_out.data = result
            self.publisher_.publish(msg_out)
            self.get_logger().info(f"Published result: {result}")

            # logging anomalies
            if self.enable_logging:
                anomalies = self._detect(result)
                if anomalies:
                    timestamp = time.time()
                    if self.start_time is None:
                        self.start_time = timestamp
                    duration = timestamp - self.start_time
                    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
                    # save anomaly image
                    img_path = f"/ros2_workspace/src/ros2_uform/ros2_uform/logging/log_image/anomaly_{ts}.png"
                    cv2.imwrite(img_path, bgr_image)
                    kinds = ";".join(anomalies)
                    with open(self.log_file, 'a') as f:
                        f.write(f"{ts},{duration:.2f},{img_path},{kinds}\n")
                    self.get_logger().info(f"Anomaly: time={ts}, dur={duration:.2f}s, type={kinds}")
                else:
                    self.start_time = None

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = UformInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
