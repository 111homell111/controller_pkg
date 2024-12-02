#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LineFollower:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('line_follower', anonymous=True)

        # Create a publisher for the /cmd_vel topic
        self.velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)

        # Subscribe to the camera feed
        rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.camera_callback)

        # Initialize the CvBridge to convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Initialize the frame
        self.frame = None

        # Stop the robot on shutdown
        rospy.on_shutdown(self.stop_robot)

    def stop_robot(self):
        """
        Function to stop the robot when the script is terminated.
        """
        rospy.loginfo("Shutting down... Stopping the robot.")
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.velocity_publisher.publish(stop_twist)
        cv2.destroyAllWindows()

    def line_following(self, frame):
        """
        Perform line-following based on the given frame where the first half of the path
        is dark grey and bordered by white lines.

        :param frame: The current image frame captured from the robot's camera.
        :return: The linear and angular velocity commands for the robot.
        """

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding to detect the dark grey path (inverted binary)
        _, path_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # Dark grey path

        # Apply binary thresholding to detect the white borders
        _, border_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # White borders

        # Find contours of the white borders
        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the center of the largest contour (white border)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            # Ensure the moment calculation is valid (avoid division by zero)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])  # X-coordinate of the center of the largest contour
            else:
                cx = frame.shape[1] // 2  # Default to the center of the image if no valid contour
        else:
            cx = frame.shape[1] // 2  # Default to the center of the image if no contours found

        # Get the center of the image (for comparison)
        image_center_x = frame.shape[1] // 2

        # Calculate the error (distance from the center of the image)
        error = cx - image_center_x  # Positive means the robot is off to the right

        # Proportional control: Adjust steering based on error
        steering_factor = 0.01  # Tuning parameter for how strongly to steer based on error
        angular_velocity = -steering_factor * error  # Negative to steer towards the path

        # Define linear velocity
        if np.sum(path_mask) > 0:  # The dark grey region has non-zero pixels
            linear_velocity = 0.6  # Move forward
        else:
            linear_velocity = 0.0  # Stop if no path is detected

        return linear_velocity, angular_velocity

    def camera_callback(self, msg):
        """
        Callback function to process frames from the camera topic.

        :param msg: The ROS Image message.
        """
        # Convert the ROS Image message to an OpenCV image
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        # Perform line-following logic if the frame is valid
        if self.frame is not None:
            linear_velocity, angular_velocity = self.line_following(self.frame)

            # Create and publish the Twist message
            twist = Twist()
            twist.linear.x = linear_velocity
            twist.angular.z = angular_velocity
            self.velocity_publisher.publish(twist)

            # Optional: Display the processed frame
            cv2.putText(self.frame, f"Linear: {linear_velocity:.2f}, Angular: {angular_velocity:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Line Following', self.frame)

            # Break on user input 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown('User requested shutdown.')

    def run(self):
        """
        Run the ROS node.
        """
        rospy.loginfo("Line Follower Node Running")
        rospy.spin()


if __name__ == '__main__':
    try:
        line_follower = LineFollower()
        line_follower.run()
    except rospy.ROSInterruptException:
        pass
