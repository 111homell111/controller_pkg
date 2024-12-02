#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from cv_bridge import CvBridge
import cv2
from tf.transformations import quaternion_from_euler
import numpy as np


# first spawn point
roll = 0.0  # rotation around X-axis
pitch = 0.0  # rotation around Y-axis
yaw = 1.57  # rotation around Z-axis (radians)
quaternion1 = quaternion_from_euler(roll, pitch, yaw)
FIRST_SPAWN = [0.5,0,0.2,quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]]

# second spawn
roll = 0.0  # rotation around X-axis
pitch = 0.0  # rotation around Y-axis
yaw = 3.14  # rotation around Z-axis (radians)
quaternion2 = quaternion_from_euler(roll, pitch, yaw)
SECOND_SPAWN = [-4,0.5,0.2,quaternion2[0],quaternion2[1],quaternion2[2],quaternion2[3]]

# third spawn
roll = 0.0  # rotation around X-axis
pitch = 0.0  # rotation around Y-axis
yaw = -0.10  # rotation around Z-axis (radians)
quaternion3 = quaternion_from_euler(roll, pitch, yaw)
THIRD_SPAWN = [-4.15,-2.3,0.2,quaternion3[0],quaternion3[1],quaternion3[2],quaternion3[3]]



def spawn_position(position):
    msg = ModelState()
    msg.model_name = 'B1'

    msg.pose.position.x = position[0]
    msg.pose.position.y = position[1]
    msg.pose.position.z = position[2]
    # in radian quaternions
    msg.pose.orientation.x = position[3]
    msg.pose.orientation.y = position[4]
    msg.pose.orientation.z = position[5]
    msg.pose.orientation.w = position[6]

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(msg)
        rospy.loginfo("Model state set successfully")
    except rospy.ServiceException:
        rospy.logerr("Service call failed")


def spawnTo_clue(clue):
    if (clue == 1):
        spawn_position(FIRST_SPAWN)
        move_robot(-4,0,0.65)
        move_robot(0,-6,0.18)
        move_robot(2,0,0.3)
        move_robot(0,0,0.2)
        move_robot(0,6,0.21)
    elif clue == 2:
        spawn_position(SECOND_SPAWN)
        move_robot(0,7,0.183)
        move_robot(-3,0, 1.2)
        move_robot(0,-6,0.38)
        move_robot(0,0,0.2)
    elif clue == 3:
        spawn_position(THIRD_SPAWN)
        # move_robot(-2, 0, 0.75)
        move_robot(0,6,0.35)
        move_robot(4, -7, 0.13)


def move_robot(linear_x, angular_z, duration):
    """
    Move the robot at the specified linear speed and angular speed for a given duration.
    """
    velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
    rospy.sleep(1)
    rate = rospy.Rate(10)
    vel_msg = Twist()
    vel_msg.linear.x = linear_x
    vel_msg.angular.z = angular_z
    rospy.loginfo(f"Car is moving with linear speed {linear_x} m/s and angular speed {angular_z} rad/s")
    start_time = rospy.Time.now()
    while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < duration:
        velocity_publisher.publish(vel_msg)
        rate.sleep()
    # Send stop command at the end of movement
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    velocity_publisher.publish(vel_msg)
    rospy.loginfo("Car has stopped")

def stop_robot():
    """Stop the robot immediately."""
    velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
    stop_msg = Twist()
    stop_msg.linear.x = 0.0
    stop_msg.angular.z = 0.0
    velocity_publisher.publish(stop_msg)
    rospy.sleep(1)
    rospy.loginfo("Robot stopped.")

def shutdown_callback():
    """Handles ROS shutdown to stop the robot."""
    stop_robot()
    rospy.loginfo("Shutting down and stopping robot.")

class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.camera_callback)
        self.bridge = CvBridge()
        self.frame = None
        # self.state = "initial_movement"  # Start with initial movement state
        self.state = "truck_teleport"
        self.start_time = rospy.Time.now()
        self.no_contour_counter = 0  
        self.clueboard_center = None  # To store the center of the detected clueboard

        self.pub_image = rospy.Publisher('/B1/rrbot/camera1/clueboard_image', Image, queue_size=1)  # For publishing image with highlighted clueboards
        

        rospy.on_shutdown(shutdown_callback)

    def end(self):
        rospy.on_shutdown(shutdown_callback)

    def truck_teleport(self):
        spawnTo_clue(1)
        self.state = "line_following"

    
    def initial_movement(self):
        """Handle the hardcoded initial movement."""
        move_robot(0, 3, 0.18)
        move_robot(2.5, 0, 0.49)
        move_robot(0, 0, 0.2)

        move_robot(0, -3, 0.35)
        move_robot(3, 0.74, 2.05)
        self.state = "line_following"  # Transition to line following after movement

    def line_following(self, frame):
        """
        Adjust robot velocity to stay within the road bounded by white lines.
        Ensures the leftmost white contour stays to the robot's left.
        Publishes an image with the guide contour highlighted.
        """
        # Convert frame to grayscale and threshold to binary
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Focus only on the bottom half of the frame
        height, width = frame.shape[:2]
        binary_bottom = binary[height // 2 :, :]

        # Find contours in the bottom half
        contours, _ = cv2.findContours(binary_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            rospy.logwarn("No contours found in the bottom half! Robot stopping.")
            return 0.0, 0.0  # Stop the robot

        # Identify the leftmost contour
        leftmost_contour = min(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

        # Compute the centroid of the selected contour
        moments = cv2.moments(leftmost_contour)
        if moments['m00'] == 0:  # Avoid division by zero
            rospy.logwarn("Degenerate contour with zero area.")
            return 0.0, 0.0  # Stop the robot
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00']) + height // 2  # Adjust for bottom-half cropping

        # Determine angular velocity based on contour's position
        angular_velocity = 0.004 * (width // 2 - cx - 350)

        # Avoid crossing the contour: increase angular velocity if the contour is very close
        bottom_of_contour = max(leftmost_contour, key=lambda point: point[0][1])[0][1] + height // 2
        if bottom_of_contour > (height - 50):  # Contour is near the bottom of the frame
            angular_velocity *= 2  # Turn more aggressively

        # Highlight the selected contour for visualization
        highlighted_frame = frame.copy()
        cv2.drawContours(highlighted_frame[height // 2 :, :], [leftmost_contour], -1, (0, 255, 0), 3)

        # Publish the frame with highlighted contours
        self.publish_frame_with_highlights(highlighted_frame, contours)

        linear_velocity = 0.2

        # Set constant linear velocity, but slow down if the robot is very close to the line
        linear_velocity = 0.4 if bottom_of_contour < (height - 50) else 0.2








        # Clueboard detection logic (commented out for now)
        # if self.detect_clueboard(frame):
        #     print("Clueboard detected! Switching state.")
        #     self.state = "approach_clueboard"
        #     return 0.0, 0.0  # Stop movement while switching state

        # Publish highlighted frame
   


        # Clueboard detection logic (debugging)
        # if self.detect_clueboard(frame):
        #     print("Clueboard detected!")
        #     # Optionally, print the size of the clueboard (bounding box area or contour area)
        #     clueboard_size = self.get_clueboard_size(frame)
        #     print(f"Clueboard size: {clueboard_size} pixels")

        #     # Set the robot state to approach the clueboard
        #     # self.state = "approaching_clueboard"
        #     self.clueboard_center = self.get_clueboard_center(frame)

        return linear_velocity, angular_velocity

    def approaching_clueboard(self, frame):
        """Approach the detected clueboard."""
        if self.clueboard_center:
            # Calculate the offset between the robot and the clueboard
            image_center_x = frame.shape[1] // 2
            error = self.clueboard_center - image_center_x + 260

            steering_factor = 0.01
            if self.clueboard_center < 100:
                steering_factor = 0.016

            angular_velocity = -steering_factor * error
            linear_velocity = 0.1  # Move towards the clueboard

            print(f"Approaching clueboard. Steering: {angular_velocity}, Linear velocity: {linear_velocity}")

            # Once the robot is close enough to the clueboard, transition back to line following
            if abs(error) < 10:  # Threshold for reaching the clueboard
                self.state = "line_following"  # Go back to line following state
                self.clueboard_center = None  # Reset clueboard center

        else:
            # If no clueboard detected (fallback), just stop
            angular_velocity = 0
            linear_velocity = 0
            print("No clueboard center available. Stopping.")

        return linear_velocity, angular_velocity

    def detect_clueboard(self, frame):
        """Detect color regions that could be clueboards, ignoring the sky."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the blue color range for clueboard detection
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Mask out the top half of the screen to ignore the sky
        height, width = mask.shape
        mask[:height//2, :] = 0  # Ignore the top half of the image

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Identify the largest clueboard
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)

            for contour in contours:
                if cv2.contourArea(contour) == largest_area:
                    # Highlight the largest contour in red
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)  # Red contour
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red bounding box
                else:
                    # Highlight other contours in green
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)  # Green contour
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green bounding box

            # Publish the frame with highlighted clueboards
            self.publish_frame_with_highlights(frame, [])

            return True
        return False


    def publish_frame_with_highlights(self, frame, contours):
        """
        Publish an image with highlighted contours.
        """
        if contours:
            for contour in contours:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)  # Green for contours
        # Convert to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_image.publish(msg)


    def get_clueboard_size(self, frame):
        """Calculate and return the size of the clueboard (area of contour)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Mask out the top half of the screen to ignore the sky
        height, width = mask.shape
        mask[:height//2, :] = 0  # Ignore the top half of the image

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If any contours are detected, calculate the area of the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            return area
        return 0

    def get_clueboard_center(self, frame):
        """Calculate and return the center of the clueboard."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Mask out the top half of the screen to ignore the sky
        height, width = mask.shape
        mask[:height//2, :] = 0  # Ignore the top half of the image

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                print(f"clueboard center: {cx}")
                return cx
               
        return None

    def camera_callback(self, msg):
        try:
            linear_velocity = 0.0
            angular_velocity = 0.0
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.state == "initial_movement":
                self.initial_movement()
            elif self.state == "line_following":
                linear_velocity, angular_velocity = self.line_following(self.frame)
            elif self.state == "end":
                linear_velocity = 0.0
                angular_velocity = 0.0
            elif self.state == "truck_teleport":
                self.truck_teleport()
            # elif self.state == "approaching_clueboard":
            #     linear_velocity, angular_velocity = self.approaching_clueboard(self.frame)

            # Publish velocity commands
            vel_msg = Twist()
            vel_msg.linear.x = linear_velocity
            vel_msg.angular.z = angular_velocity
            self.velocity_publisher.publish(vel_msg)

        except Exception as e:
            rospy.logerr(f"Error in camera callback: {e}")

    def shutdown_callback(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)
        rospy.loginfo("Shutting down the line follower node.")




if __name__ == '__main__':
    try:
        line_follower = LineFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.on_shutdown(shutdown_callback)
        pass
    except KeyboardInterrupt:
        rospy.on_shutdown(shutdown_callback)
