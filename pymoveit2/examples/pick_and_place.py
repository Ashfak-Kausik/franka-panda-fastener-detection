#!/usr/bin/env python3
"""
YOLOv12-guided fastener pick and place node.
Detects Clip, Rivet, Screw using YOLOv12 and sorts into separate bins.

ros2 run pymoveit2 pick_and_place.py --ros-args -p target_color:=Clip
ros2 run pymoveit2 pick_and_place.py --ros-args -p target_color:=Rivet
ros2 run pymoveit2 pick_and_place.py --ros-args -p target_color:=Screw
"""

from threading import Thread
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

from pymoveit2 import MoveIt2, GripperInterface
from pymoveit2.robots import panda

import math


class PickAndPlace(Node):
    def __init__(self):
        super().__init__("pick_and_place")

        # Parameters
        self.declare_parameter("target_color", "Clip")
        self.target_color = self.get_parameter("target_color").value.strip()

        self.declare_parameter("approach_offset", 0.31)
        self.approach_offset = float(
            self.get_parameter("approach_offset").value
        )

        # Flags
        self.already_moved = False
        self.target_coords = None

        self.callback_group = ReentrantCallbackGroup()

        # Arm MoveIt2 interface
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )

        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1

        # Gripper interface
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=panda.MOVE_GROUP_GRIPPER,
            callback_group=self.callback_group,
            gripper_command_action_name="gripper_action_controller/gripper_cmd",
        )

        # Subscriber
        self.sub = self.create_subscription(
            String, "/color_coordinates", self.coords_callback, 10
        )
        self.get_logger().info(
            f"Waiting for {self.target_color} from /color_coordinates..."
        )

        # Joint positions
        self.start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.radians(-125.0)]
        self.home_joints  = [0.0, 0.0, 0.0, math.radians(-90.0), 0.0,
                             math.radians(92.0), math.radians(50.0)]

        # Three separate drop positions — one per fastener type
        # These use the same base drop position for now
        # You can tune each one later once simulation is running
        self.drop_joints = {
            "Clip": [
                math.radians(-155.0), math.radians(30.0), math.radians(-20.0),
                math.radians(-124.0), math.radians(44.0), math.radians(163.0),
                math.radians(7.0)
            ],
            "Rivet": [
                math.radians(-130.0), math.radians(30.0), math.radians(-20.0),
                math.radians(-124.0), math.radians(44.0), math.radians(163.0),
                math.radians(7.0)
            ],
            "Screw": [
                math.radians(-105.0), math.radians(30.0), math.radians(-20.0),
                math.radians(-124.0), math.radians(44.0), math.radians(163.0),
                math.radians(7.0)
            ],
        }

        # Move to start
        self.moveit2.move_to_configuration(self.start_joints)
        self.moveit2.wait_until_executed()

    def coords_callback(self, msg):
        if self.already_moved:
            return

        try:
            parts = msg.data.split(",")
            fastener_id = parts[0].strip()   # 'Clip', 'Rivet', or 'Screw'
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            if fastener_id == self.target_color:
                self.target_coords = [x, y, z]
                self.get_logger().info(
                    f"Target {self.target_color} locked at: "
                    f"[{x:.3f}, {y:.3f}, {z:.3f}]"
                )
                self.already_moved = True

                pick_position = [x, y, z - 0.60]
                quat_xyzw = [0.0, 1.0, 0.0, 0.0]

                # Get the correct drop bin for this fastener
                drop = self.drop_joints.get(self.target_color, self.drop_joints["Clip"])

                # --- Pick and place sequence ---

                # 1. Move to home
                self.moveit2.move_to_configuration(self.home_joints)
                self.moveit2.wait_until_executed()

                # 2. Move above target
                self.moveit2.move_to_pose(
                    position=pick_position, quat_xyzw=quat_xyzw)
                self.moveit2.wait_until_executed()

                # 3. Open gripper
                self.gripper.open()
                self.gripper.wait_until_executed()

                # 4. Move down to object
                approach_position = [
                    pick_position[0],
                    pick_position[1],
                    pick_position[2] - self.approach_offset
                ]
                self.moveit2.move_to_pose(
                    position=approach_position,
                    quat_xyzw=quat_xyzw,
                    cartesian=True
                )
                self.moveit2.wait_until_executed()

                # 5. Close gripper (grasp)
                self.gripper.close()
                self.gripper.wait_until_executed()

                # 6. Return to home
                self.moveit2.move_to_configuration(self.home_joints)
                self.moveit2.wait_until_executed()

                # 7. Move to correct bin for this fastener type
                self.moveit2.move_to_configuration(drop)
                self.moveit2.wait_until_executed()

                # 8. Release
                self.gripper.open()
                self.gripper.wait_until_executed()

                # 9. Close gripper
                self.gripper.close()
                self.gripper.wait_until_executed()

                # 10. Return to start
                self.moveit2.move_to_configuration(self.start_joints)
                self.moveit2.wait_until_executed()

                self.get_logger().info(
                    f"Pick-and-place complete: {self.target_color} → bin"
                )
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error parsing /color_coordinates: {e}")


def main():
    rclpy.init()
    node = PickAndPlace()

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        executor_thread.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
