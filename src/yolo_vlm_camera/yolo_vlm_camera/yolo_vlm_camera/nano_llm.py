#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PIL_Image
from nano_llm import NanoLLM, ChatHistory

class NanoLLMNode(Node):
    def __init__(self):
        super().__init__('nano_llm')
        # Parameters (can be provided via launch file or parameter file)
        self.declare_parameter('model', "Efficient-Large-Model/VILA-2.7b")
        self.declare_parameter('api', "mlc")
        self.declare_parameter('quantization', "q4f16_ft")
      
        # Subscribe to the "input_image" topic for image messages
        self.image_subscription = self.create_subscription(
            Image,
            'input_image',
            self.image_callback,
            10)
        self.cv_br = CvBridge()

        # Create a publisher to publish VLM output on the "output" topic
        self.output_publisher = self.create_publisher(String, 'output', 10)
        
        # Load the VLM model and initialize chat history
        self.model = NanoLLM.from_pretrained(self.get_parameter('model').value)
        self.chat_history = ChatHistory(self.model)
        
    def image_callback(self, msg: Image): 
        self.get_logger().info("Received input_image for VLM inference.")
        cv_img = self.cv_br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        pil_img = PIL_Image.fromarray(cv_img)
        
        prompt = "Describe the image."
        self.get_logger().info("Running VLM inference on received image.")
        
        # Append the image and text to the chat history and perform inference
        self.chat_history.append('user', image=pil_img)
        self.chat_history.append('user', prompt, use_cache=True)
        embedding, _ = self.chat_history.embed_chat()
      
        output = self.model.generate(
            inputs=embedding,
            kv_cache=self.chat_history.kv_cache,
            min_new_tokens=10,
            streaming=False, 
            do_sample=True,
        )
        self.get_logger().info(f"VLM output: {output}")

        # Publish the generated output to the "output" topic
        output_msg = String()
        output_msg.data = output
        self.output_publisher.publish(output_msg)
        
        self.chat_history.reset()
        
def main(args=None):
    rclpy.init(args=args)
    node = NanoLLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down VLM node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
