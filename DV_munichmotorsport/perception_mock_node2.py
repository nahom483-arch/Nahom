from dataclasses import dataclass
import time
import rclpy
from rclpy.node import Node
from rclpy.time import Duration, Time
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from tf2_ros import TransformException
from tf2_ros import TransformStamped
import tf2_ros
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TwistStamped
from tf2_ros.buffer import Buffer
import math
import numpy as np
from sklearn.cluster import DBSCAN
from dv_ros_msgs.msg import LocalCones, Cone, StartFinishLine
from dv_ros_msgs.msg import Map
from dv_ros_msgs.srv import MapSrv
import math
from dv_test_map_provider.map_srv import MapProfile
import dv_integration_test_perception_mock.quatmath as quatmath
from std_msgs.msg import Header
import random
from dv_ros_msgs.srv import PlotterAddMap, PlotterSave, PlotterAddVehicleState
import threading
import copy

# For Pipeline timing
from sensor_msgs.msg import PointCloud2

# Probability of missing a visible cone (e.g., 5% miss rate)
MISS_PROBABILITY = 0.05
DISTURBANCE_LOCAL_CONES = 0.05 # goes in x and y direction
DISTURBANCE_MAP_CONES = 0.05  # goes in x and y direction
CONE_PIPELINE_DELAY = 0.1 # estimated pipeline delay from lidar to local_cones in s
SHOW_INVISIBLE_CONES = True
VEHICLE_LIDAR = 0.8 # ~ conservative estimation (use tf frames!!!)


@dataclass
class VehivlePosition:
    x: float
    y: float


class TestNode(Node):
    def __init__(self):
        
        self.vehiclestatecount = 0
        self.max_vehiclestates = 4
        self.vehiclestate_in = None
        self.vehiclestates = [None, None, None, None]
        self.vehicle_position: VehivlePosition = None
        self.start_finish_line_cones = None

        super().__init__("perception_mock_node")
        self.local_cone_pub = self.create_publisher(LocalCones, "local_cones", 1)
        sfline_topic_name = self.declare_parameter('start_finish_line_topic_name', 'start_finish_line').get_parameter_value().string_value
        self.sf_line_pub = self.create_publisher(StartFinishLine, sfline_topic_name, 1)
        vehiclestate_topic_name = self.declare_parameter('vehiclestate_topic_name', 'vehiclestate').get_parameter_value().string_value
        self.vehiclestate_sub = self.create_subscription(TwistStamped, vehiclestate_topic_name, callback=self.vehiclestate_callback, qos_profile=1)
        self.map_transform_id = self.declare_parameter('map_transform_id', 'world').get_parameter_value().string_value##world

        # For Pipeline testing
        self.rslidar_points_pub = self.create_publisher(PointCloud2, 'rslidar_points', 1)
        self.lidar_nonground_pub = self.create_publisher(PointCloud2, 'lidar_nonground', 1)
        self.local_cones_no_color_msg_pub = self.create_publisher(LocalCones, 'local_cones_no_color', 1)

        self.tf_buffer = Buffer(Duration(seconds=5))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timestep = 0.1 # lidar publishing frequency
        self.lidar_timer = self.create_timer(self.timestep, self.lidar_callback)

        self.perception_distance = 24#1.6 + 13

        self.global_map = Map()
        self.local_cones = LocalCones()
        self.init_map() 

        self.get_logger().info(f"LIDAR_OFFSET: {VEHICLE_LIDAR}") # Log   ...LIDAR_OFFSET


    def init_map(self):
        map_client = self.create_client(MapSrv, 'map_service')
        while not map_client.wait_for_service(timeout_sec=15.0):#3.0
            self.get_logger().info('Waiting for map provider...')
        map_req = MapSrv.Request()
        map_req.map_profile = MapProfile.TIMING_TRACK_1.value
        # map_req.map_profile = MapProfile.TRACK_1_ACCELL.value
        # map_req.map_profile = MapProfile.TRACK_2_SKIDPAD.value
        # map_req.map_profile = MapProfile.VSV_TRACK.value
        map_req.disturbance_max = DISTURBANCE_MAP_CONES
        map_future = map_client.call_async(map_req)
        self.get_logger().info("Requesting Map...")
        rclpy.spin_until_future_complete(self, map_future)
        self.get_logger().info("Received map!")        
        self.global_map = map_future.result().map_response
        self.start_finish_line_cones = self.calc_start_finish_lines()
        self.get_logger().info(f"Detected amount of start finish lines: {len(self.start_finish_line_cones)}")
        # plot map
        if False:
            self.add_map_client = self.create_client(PlotterAddMap, 'add_map')
            self.save_client = self.create_client(PlotterSave, 'save_svg')

            while not self.add_map_client.wait_for_service(timeout_sec=3.0):
                self.get_logger().info('Waiting for add_map service...')
            while not self.save_client.wait_for_service(timeout_sec=3.0):
                self.get_logger().info('Waiting for save service...')
            
            self.add_map_req = PlotterAddMap.Request()        
            self.save_req = PlotterSave.Request()
            self.add_map_req.plot_name = 'test'         
            self.add_map_req.map = self.global_map
            self.add_map_req.plot_blue_cones = True
            self.add_map_req.plot_yellow_cones = True
            self.add_map_req.plot_centerline = True
            self.add_map_req.plot_triangulation = True        

            add_future = self.add_map_client.call_async(self.add_map_req)
            rclpy.spin_until_future_complete(self, add_future)

            self.save_req.plot_name = 'test'
            self.save_req.path = './plots/'
            save_future = self.save_client.call_async(self.save_req)

            self.get_logger().info("Requesting save!")
            rclpy.spin_until_future_complete(self, save_future)
    
    def calc_start_finish_lines(self):
        """
        Calculate all potential start/finish lines based on clusters of orange cones.
        Returns a list of finish lines (each line being a pair of cones).
        """
        # Filter orange cones
        large_orange_cones = list(filter(lambda cone: cone.type == Cone.ORANGE_LARGE, self.global_map.cones))
        
        if len(large_orange_cones) < 2:
            self.get_logger().info(f"Error! Not enough orange cones ({len(large_orange_cones)}) to form finish lines.")
            return []

        # Prepare cone positions for clustering
        cone_positions = np.array([[cone.position.x, cone.position.y] for cone in large_orange_cones])
        
        # Cluster cones using DBSCAN (Density-Based Spatial Clustering)
        # eps: maximum distance between two samples to be considered in the same neighborhood
        # min_samples: minimum number of samples in a neighborhood to form a cluster
        clustering = DBSCAN(eps=4.0, min_samples=2).fit(cone_positions)
        labels = clustering.labels_
        
        finish_lines = []
        
        # Process each cluster
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_cones = [large_orange_cones[i] for i, label in enumerate(labels) if label == cluster_id]
            
            if len(cluster_cones) == 2:
                # Directly use the pair as a finish line
                finish_lines.append(cluster_cones)
            elif len(cluster_cones) == 4:
                # Find the two most separated pairs in this cluster
                # First find the two most distant cones
                min_dist = math.inf
                c1, c2 = None, None
                for i in range(len(cluster_cones)):
                    for j in range(i+1, len(cluster_cones)):
                        dist = math.hypot(cluster_cones[i].position.x - cluster_cones[j].position.x,
                                        cluster_cones[i].position.y - cluster_cones[j].position.y)
                        if dist < min_dist:
                            min_dist = dist
                            c1, c2 = cluster_cones[i], cluster_cones[j]
                
                # The remaining two cones should form the other pair
                remaining = [cone for cone in cluster_cones if cone not in (c1, c2)]
                if len(remaining) == 2:
                    # Create virtual cones at the midpoints
                    cone1, cone2 = Cone(), Cone()
                    cone1.position.x = (c1.position.x + c2.position.x) / 2
                    cone1.position.y = (c1.position.y + c2.position.y) / 2
                    cone2.position.x = (remaining[0].position.x + remaining[1].position.x) / 2
                    cone2.position.y = (remaining[0].position.y + remaining[1].position.y) / 2
                    finish_lines.append([cone1, cone2])
        
        if not finish_lines:
            self.get_logger().info("No valid finish lines could be identified from orange cones.")
        else:
            for i, line in enumerate(finish_lines):
                self.get_logger().info(f"Finish line no:{i+1} identified with points: ({(line[0].position.x):0.2f};{(line[0].position.y):0.2f}) and ({(line[1].position.x):0.2f};{(line[1].position.y):0.2f})")

        
        return finish_lines
    
    def cone_in_range(self, x: float, y: float, angle1: float, angle2: float):
        if ((math.pow(x,2) + math.pow(y,2)) < self.perception_distance*self.perception_distance):
            ca = math.atan2(y, x)
            if(angle1 < angle2):
                if(ca > angle1 and ca < angle2):
                    return True
            else:
                if(ca > angle1 or ca < angle2):
                    return True                
        return False


    # def build_local_map(self, transform_vehicle: TransformStamped):
    #     self.local_cones = LocalCones()
    #     self.local_cones.header.stamp = transform_vehicle.header.stamp
    #     self.local_cones.header.frame_id = "vehicle"

    #     for cone in self.global_map.cones:
    #         # 1. Express cone position in map frame as PointStamped
    #         p_map = PointStamped()
    #         p_map.header.frame_id = self.map_transform_id
    #         p_map.header.stamp = transform_vehicle.header.stamp
    #         p_map.point.x = cone.position.x
    #         p_map.point.y = cone.position.y
    #         p_map.point.z = 0.0

    #         try:
    #             # 2. Transform map → lidar
    #             p_lidar = self.tf_buffer.transform(p_map, "lidar", timeout=Duration(seconds=0.1))
    #         except TransformException as ex:
    #             self.get_logger().warn(f"Transform map→lidar failed: {ex}")
    #             continue

    #         # 3. Do visibility check in lidar frame
    #         if self.cone_in_range(p_lidar.point.x, p_lidar.point.y, -math.pi/3, math.pi/3):
    #             if random.random() < MISS_PROBABILITY:
    #                 continue

    #             # add noise in lidar frame
    #             p_lidar.point.x += random.uniform(-DISTURBANCE_LOCAL_CONES, DISTURBANCE_LOCAL_CONES)
    #             p_lidar.point.y += random.uniform(-DISTURBANCE_LOCAL_CONES, DISTURBANCE_LOCAL_CONES)

    #             try:
    #                 # 4. Transform lidar → vehicle
    #                 p_vehicle = self.tf_buffer.transform(p_lidar, "vehicle", timeout=Duration(seconds=0.1))
    #             except TransformException as ex:
    #                 self.get_logger().warn(f"Transform lidar→vehicle failed: {ex}")
    #                 continue

    #             # 5. Store in message
    #             transcone = Cone()
    #             transcone.position.x = p_vehicle.point.x
    #             transcone.position.y = p_vehicle.point.y
    #             transcone.type = cone.type
    #             self.local_cones.cones.append(transcone)
    
    
    
    # def build_local_map(self, transform_vehicle: TransformStamped):
    #     """
    #     Build a local cone map from global cones.
    #     1. Transform cones into lidar frame
    #     2. Apply FOV / range filtering
    #     3. Transform back to vehicle frame for publishing
    #     """

    #     # Get transform vehicle->lidar (static, known from URDF or TF)
    #     try:
    #         tf_vehicle_to_lidar = self.tf_buffer.lookup_transform(
    #             target_frame="lidar",    # your lidar frame id (check in TF tree!)
    #             source_frame="vehicle",
    #             time=rclpy.time.Time(),
    #             timeout=Duration(seconds=0.1)
    #         )
    #     except TransformException as ex:
    #         self.get_logger().warn(f"Could not get transform vehicle->lidar: {ex}")
    #         return

    #     # also invert it (lidar->vehicle) since we’ll need it for publishing
    #     try:
    #         tf_lidar_to_vehicle = self.tf_buffer.lookup_transform(
    #             target_frame="vehicle",
    #             source_frame="lidar",
    #             time=rclpy.time.Time(),
    #             timeout=Duration(seconds=0.1)
    #         )
    #     except TransformException as ex:
    #         self.get_logger().warn(f"Could not get transform lidar->vehicle: {ex}")
    #         return

    #     # vehicle pose in map frame (from SLAM or ground truth)
    #     _, _, yaw = quatmath.quaternion_to_euler(transform_vehicle.transform.rotation)

    #     self.local_cones = LocalCones()
    #     h = Header()
    #     h.stamp = transform_vehicle.header.stamp
    #     h.frame_id = "vehicle"
    #     self.local_cones.header = h

    #     for cone in self.global_map.cones:

    #         # --- Step 1: transform global cone -> lidar frame ---
    #         try:
    #             point_in_lidar = tf2_geometry_msgs.do_transform_point(cone.position, 
    #                                                                 transform_vehicle)  
    #             # note: you may need `tf2_geometry_msgs` for PointStamped
    #         except Exception as e:
    #             self.get_logger().warn(f"TF transform failed: {e}")
    #             continue

    #         # --- Step 2: apply sensor FOV and perception distance ---
    #         x, y = point_in_lidar.point.x, point_in_lidar.point.y
    #         if self.cone_in_range(x, y, -math.pi/3, math.pi/3):  # 120° FOV
    #             if random.random() < MISS_PROBABILITY:
    #                 continue  # simulate missed detection

    #             # add noise in lidar frame
    #             x += random.uniform(-DISTURBANCE_LOCAL_CONES, DISTURBANCE_LOCAL_CONES)
    #             y += random.uniform(-DISTURBANCE_LOCAL_CONES, DISTURBANCE_LOCAL_CONES)

    #             # --- Step 3: transform back lidar->vehicle ---
    #             # easiest: just apply the tf_lidar_to_vehicle to (x,y,0)
    #             local_x = x * np.cos(yaw) + y * np.sin(yaw)
    #             local_y = -x * np.sin(yaw) + y * np.cos(yaw)

    #             transcone = Cone()
    #             transcone.position.x = local_x
    #             transcone.position.y = local_y
    #             transcone.type = cone.type
    #             self.local_cones.cones.append(transcone)

    
    
    def build_local_map(self, transform: TransformStamped):
        _, _, yaw = quatmath.quaternion_to_euler(transform.transform.rotation)
        angle1 = yaw - (math.pi/3)
        if(angle1 < -math.pi):
            angle1 = angle1 + (2*math.pi)

        angle2 = yaw + (math.pi/3)
        if(angle2 > math.pi):
            angle2 = angle2 - (2* math.pi)


        self.local_cones = LocalCones()
        h: Header=Header()
        h.stamp = transform.header.stamp
        self.local_cones.header = h
        self.local_cones.header.frame_id = 'vehicle'

        for cone in self.global_map.cones:
            transcone = Cone()
            x = cone.position.x - transform.transform.translation.x
            y = cone.position.y - transform.transform.translation.y#  - VEHICLE_LIDAR # dirty vehicle to lidar TODO: do properly!


            if math.sqrt(x**2 + y**2) < self.perception_distance:
                # always calculate in vehicle frame
                if self.cone_in_range(x-math.cos(yaw) * 2, y-math.sin(yaw) * 2, angle1, angle2):

                    # Randomly skip cones to simulate missed detection
                    if random.random() < MISS_PROBABILITY:
                        continue  # Skip this cone entirely

                    local_x = x * np.cos(yaw) + y * np.sin(yaw)
                    local_y = -x * np.sin(yaw) + y * np.cos(yaw)#  + VEHICLE_LIDAR # dirty lidar to vehicle TODO: do properly!

                    # Add noise to the adjusted local positions
                    transcone.position.x = local_x + random.uniform(-DISTURBANCE_LOCAL_CONES, DISTURBANCE_LOCAL_CONES)
                    transcone.position.y = local_y + random.uniform(-DISTURBANCE_LOCAL_CONES, DISTURBANCE_LOCAL_CONES)
                    transcone.type = cone.type
                    self.local_cones.cones.append(transcone)
            
                elif SHOW_INVISIBLE_CONES:
                    # add "invisible" cones surrounding the vehicle to the local map in green
                    transcone.position.x = x * np.cos(yaw) + y * np.sin(yaw)
                    transcone.position.y = -x * np.sin(yaw) + y * np.cos(yaw)
                    transcone.type = Cone.UNCLASSIFIED
                    self.local_cones.cones.append(transcone)
    
    def vehiclestate_callback(self, vehiclestate: TwistStamped):
        # update new vehicle state
        self.vehiclestate_in = vehiclestate

        def ccw(A, B, C):
            """Check if three points are in counter-clockwise order."""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            """Check if line segments AB and CD intersect."""
            # self.get_logger().info("Checking if Intersects: ")
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_transform_id,
                self.vehiclestate_in.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
        except TransformException as ex:
            self.get_logger().info(f'could not transform {self.vehiclestate_in.header.frame_id} to {self.map_transform_id}:\n{ex}')
            return

        # Extract vehicle positions
        if self.vehicle_position is None:
            self.vehicle_position = VehivlePosition(transform.transform.translation.x, transform.transform.translation.y)
            return
        
        car_previous = (self.vehicle_position.x, self.vehicle_position.y)
        car_current = (transform.transform.translation.x, transform.transform.translation.y)

        # Check if the car's movement line intersects the finish line
        for sfl_cones in self.start_finish_line_cones:
            cone_1 = (sfl_cones[0].position.x, sfl_cones[0].position.y)
            cone_2 = (sfl_cones[1].position.x, sfl_cones[1].position.y)
            # self.get_logger().info(f"{cone_1}, {cone_2}")
            # self.get_logger().info(f"{car_previous}, {car_current}")

            if intersect(cone_1, cone_2, car_previous, car_current):
                sf_line_msg: StartFinishLine = StartFinishLine()
                # Set current timestamp into header of message
                sf_line_msg.header.stamp = self.get_clock().now().to_msg()
                sf_line_msg.is_from_test_map = True
                self.sf_line_pub.publish(sf_line_msg)
                self.get_logger().info("Start/Finish line crossed!")

        # Update vehicle position
        self.vehicle_position.x = car_current[0]
        self.vehicle_position.y = car_current[1]

    def lidar_callback(self):

        def worker(local_cones):
            """Publish a LocalCones message every ~100 ms (80–120 ms)."""
            delay_duration = random.normalvariate(CONE_PIPELINE_DELAY, CONE_PIPELINE_DELAY * 0.2)  # seconds
            delay_duration = max(CONE_PIPELINE_DELAY * 0.5, min(CONE_PIPELINE_DELAY * 2, delay_duration))  # clamp between half and double the pipeline delay (shorter caluclations are less likely)
            perception_pipeline_steps = 3 #preprocessing, colordetection, conedetection

            ###############
            # For Pipeline timing: publish a dummy PointCloud2 message to simulate LiDAR input
            lidar_msg = PointCloud2()
            lidar_msg.header.stamp = local_cones.header.stamp
            self.rslidar_points_pub.publish(lidar_msg)
            self.get_logger().debug(f"Published PointCloud2 for LiDAR simulation")

            time.sleep(delay_duration/perception_pipeline_steps) #simulate preprocessing time

            lidar_nonground_msg = PointCloud2()
            lidar_nonground_msg.header.stamp = local_cones.header.stamp
            self.lidar_nonground_pub.publish(lidar_nonground_msg)
            self.get_logger().debug(f"Published PointCloud2 for LiDAR-Preprocessing simulation")

            time.sleep(delay_duration/perception_pipeline_steps) #simulate colordetection time

            local_cones_no_color_msg = LocalCones()
            local_cones_no_color_msg.header.stamp = local_cones.header.stamp
            self.local_cones_no_color_msg_pub.publish(local_cones_no_color_msg)
            self.get_logger().debug(f"Published LocalCones for Color-Detection simulation")

            time.sleep(delay_duration/perception_pipeline_steps) #simulate conedetection time
            ###############
            
            self.local_cone_pub.publish(local_cones)
            self.get_logger().debug(f"Published LocalCones after {delay_duration:.3f}s delay")

        self.vehiclestatecount = (self.vehiclestatecount+1) % self.max_vehiclestates
        self.vehiclestates[self.vehiclestatecount] = self.vehiclestate_in
        state = self.vehiclestates[self.vehiclestatecount-(self.max_vehiclestates-1)]
        if state == None: return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_transform_id,
                state.header.frame_id,
                state.header.stamp,
                timeout= Duration(nanoseconds=900000000))
                #Time())#Time.from_msg(vehiclestate.header.stamp) - Time(seconds=2, clock_type=ClockType.ROS_TIME))
        except TransformException as ex:
            self.get_logger().info(f'could not transform {self.vehiclestate_in.header.frame_id} to {self.map_transform_id}:\n{ex}')
            return
        
        if len(self.global_map.cones) > 0:
            self.build_local_map(transform)
            self.local_cones.is_simulated = True
            local_cones_copy = copy.deepcopy(self.local_cones)

            # fire off a background thread for this message
            threading.Thread(target=worker, args=(local_cones_copy,), daemon=True).start()


def main(args=None):
    rclpy.init()
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
