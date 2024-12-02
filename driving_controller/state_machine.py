#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def move_robot(linear_x, angular_z, duration):
    """
    Move the robot at the specified linear speed and angular speed for a given duration.
    """
    velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
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
        self.state = "initial_movement"  # Start with initial movement state
        self.start_time = rospy.Time.now()
        self.no_contour_counter = 0  
        self.clueboard_center = None  # To store the center of the detected clueboard

        self.pub_image = rospy.Publisher('/B1/rrbot/camera1/clueboard_image', Image, queue_size=10)  # For publishing image with highlighted clueboards
        
        rospy.on_shutdown(shutdown_callback)

    def initial_movement(self):
        """Handle the hardcoded initial movement."""
        move_robot(0, 3, 0.18)
        move_robot(2.5, 0, 0.49)
        move_robot(0, 0, 0.2)

        move_robot(0, -3, 0.35)
        move_robot(3, 0.74, 2.05)
        self.state = "line_following"  # Transition to line following after movement

    def line_following(self, frame):
        """Line following logic."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, path_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # Dark grey path
        _, border_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # White borders
        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            print(f"Contours detected: {len(contours)}")

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                print(f"Contour center x: {cx}")
            else:
                cx = frame.shape[1] // 2
                print("No contours detected.")

            # Reset the no_contour_counter when contour is detected
            self.no_contour_counter = 0

            image_center_x = frame.shape[1] // 2
            error = cx - image_center_x + 260
            steering_factor = 0.01
            if cx < 100:
                steering_factor = 0.016

            angular_velocity = -steering_factor * error

            if np.sum(path_mask) > 0:
                linear_velocity = 0.2
            else:
                linear_velocity = 0.0

        else:
            self.no_contour_counter += 1
            if self.no_contour_counter > 5:
                linear_velocity = 0.0
                angular_velocity = 0.6
            else:
                linear_velocity = 0.0
                angular_velocity = 0.0

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
            self.publish_frame_with_highlights(frame)

            return True
        return False


    def publish_frame_with_highlights(self, frame):
        """Publish the frame with highlighted clueboards to a ROS topic."""
        try:
            # Convert the OpenCV frame to a ROS image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.pub_image.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"Error publishing image: {e}")

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
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.state == "initial_movement":
                self.initial_movement()
            elif self.state == "line_following":
                linear_velocity, angular_velocity = self.line_following(self.frame)
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
        pass
