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
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    velocity_publisher.publish(vel_msg)
    rospy.loginfo("Car has stopped")

class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.camera_callback)
        self.bridge = CvBridge()
        self.frame = None
        self.state = "move_forward"  # Initial state: move forward before line following
        self.start_time = rospy.Time.now()

    def line_following(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create masks for the dark grey path (dark regions) and the white borders (bright regions)
        _, path_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # Dark grey path
        _, border_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # White borders
        
        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
            else:
                cx = frame.shape[1] // 2  # Default to center if no contours found
        else:
            cx = frame.shape[1] // 2

        image_center_x = frame.shape[1] // 2
        
        # Modify error to shift the robot to the left of the line
        error = cx - image_center_x - 50  # The -50 bias makes the robot stay left of the line
        steering_factor = 0.005
        angular_velocity = -steering_factor * error

        # If there's enough dark grey area, move forward; otherwise, stop
        if np.sum(path_mask) > 0:
            linear_velocity = 0.2  # Forward speed
        else:
            linear_velocity = 0.0  # Stop if no path detected

        return linear_velocity, angular_velocity


    def camera_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        if self.frame is not None:
            if self.state == "move_forward":
                # Drive to the first clue
                move_robot(0, 3, 0.18)
                move_robot(2.5, 0, 0.49)
                move_robot(0, 0, 0.2)

                # Drive to the second clue
                move_robot(0, -3, 0.35)
                move_robot(3, 0.739, 2.09)
                self.state = "line_following"  # Transition to line following after movement
            elif self.state == "line_following":
                # Use the line-following algorithm after moving forward
                linear_velocity, angular_velocity = self.line_following(self.frame)
                twist = Twist()
                twist.linear.x = linear_velocity
                twist.angular.z = angular_velocity
                self.velocity_publisher.publish(twist)

    def run(self):
        rospy.loginfo("Line Follower Node Running")
        rospy.spin()

if __name__ == '__main__':
    try:
        line_follower = LineFollower()
        line_follower.run()
    except rospy.ROSInterruptException:
        pass