import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
from nano_llm import NanoLLM, ChatHistory
import cv2
import os

class NanoLLMSubscriber(Node):
    def __init__(self):
        super().__init__('nano_llm_subscriber')
        # enable/disable logging
        self.declare_parameter('enable_logging', False)
        self.enable_logging = self.get_parameter('enable_logging').get_parameter_value().bool_value
        self.add_on_set_parameters_callback(self._on_param_update)

        # log file
        now = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        self.log_file = f"/ros2_workspace/src/ros2_nanollm/ros2_nanollm/logging/log_csv/log_{ts}.csv"

        log_dir = os.path.dirname(self.log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        try:
            with open(self.log_file, 'x') as f:
                f.write("timestamp,duration_s,image_path,anomaly_type\n")
        except FileExistsError:
            pass

        # load model
        self.declare_parameter('model', "Efficient-Large-Model/VILA-2.7b")
        model_name = self.get_parameter('model').get_parameter_value().string_value
        self.model = NanoLLM.from_pretrained(model_name)
        self.chat_history = ChatHistory(self.model)

        # subscriptions & publisher
        self.create_subscription(String, 'input_query', self._on_query, 10)
        self.create_subscription(Image, 'input_image', self._on_image, 10)
        self.output_pub = self.create_publisher(String, 'output', 10)

        self.bridge = CvBridge()
        self.query = "You are an AI security assistant monitoring. Analyze the given image and determine if any dangerous situations are occurring. Categories to check: Fight (fights, bullying, aggressive behavior), Fire, Smoke, Fallen person (a person falling or collapsing suddenly), Weapon possession (knives, guns, dangerous objects), General (if the scene is safe and there are no threats). Provide the response in the following format: 'Category | Description'. Ensure that the category and description match correctly based on the image. Now, analyze the following image and provide a similar response."
        self.start_time = None
        self.keywords = ['fight', 'fire', 'smoke', 'fallen person', 'weapon possession']

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'enable_logging' and p.type_ == Parameter.Type.BOOL:
                self.enable_logging = p.value
                self.get_logger().info(f"enable_logging: {self.enable_logging}")
        return SetParametersResult(successful=True)

    def _on_query(self, msg):
        self.query = msg.data

    def _detect(self, text):
        t = text.lower()
        return [kw for kw in self.keywords if kw in t]

    def _on_image(self, msg):
        now = time.time()
        if hasattr(self, 'last_time'):
            fps = 1.0/(now - self.last_time) if now != self.last_time else 0
            self.get_logger().info(f"FPS: {fps:.1f}")
        self.last_time = now

        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        pil = PILImage.fromarray(img)
        self.chat_history.append('user', image=pil)
        self.chat_history.append('user', self.query, use_cache=True)
        emb, _ = self.chat_history.embed_chat()
        out = self.model.generate(
            inputs=emb,
            kv_cache=self.chat_history.kv_cache,
            min_new_tokens=10,
            streaming=False,
            do_sample=True
        )
        self.output_pub.publish(String(data=out))
        self.get_logger().info(f"Model: {out}")

        if self.enable_logging:
            anomalies = self._detect(out)
            if anomalies:
                if self.start_time is None:
                    self.start_time = now
                dur = now - self.start_time
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
                path = f"/ros2_workspace/src/ros2_nanollm/ros2_nanollm/logging/log_image/anomaly_{ts}.png"
                cv2.imwrite(path, img)
                kinds = ";".join(anomalies)
                with open(self.log_file, 'a') as f:
                    f.write(f"{ts},{dur:.2f},{path},{kinds}\n")
                self.get_logger().info(f"Anomaly: time={ts}, dur={dur:.2f}s, types={kinds}")
            else:
                self.start_time = None
        self.chat_history.reset()


def main(args=None):
    rclpy.init(args=args)
    node = NanoLLMSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
