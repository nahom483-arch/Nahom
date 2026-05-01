import rclpy
from rclpy.node import Node
from dv_ros_msgs.msg import LocalCones
import math
import signal

class MaxDistanceNode(Node):
    def __init__(self):
        super().__init__('max_distance_node')
        self.sub = self.create_subscription(LocalCones, '/local_cones_no_color', self.callback, 10)
        self.max_dist = 0.0
        self.max_x = 0.0
        self.total_cones = 0
        self.get_logger().info("Monitoring /local_cones_no_color – press Ctrl+C to stop and see max distance")

    def callback(self, msg):
        for cone in msg.cones:
            d = math.hypot(cone.position.x, cone.position.y)
            if d > self.max_dist:
                self.max_dist = d
                self.max_x = cone.position.x
            self.total_cones += 1

    def report(self):
        if self.total_cones == 0:
            self.get_logger().info("No cones received.")
        else:
            self.get_logger().info(f"Total cones processed: {self.total_cones}")
            self.get_logger().info(f"Maximum perception distance: {self.max_dist:.2f} m (x = {self.max_x:.2f} m)")

def main():
    rclpy.init()
    node = MaxDistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()