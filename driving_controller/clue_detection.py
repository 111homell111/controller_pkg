#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import queue


def image_callback(msg):
    """
    Callback to process images from the camera topic.
    """
    try:
        print("image callback")
        # Convert the ROS Image message to OpenCV format
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        clueboard = match_and_warp(frame)
        
        cv2.imshow("Camera Feed", frame)
        if clueboard is not None:
            cv2.imshow("Cropped Clueboard", clueboard)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown('Quit')

    except CvBridgeError as e:
        rospy.logerr(f"Error converting image: {e}")

def match_and_warp(frame):
    # Load the template and target images
    template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if template_image is None or frame is None:
      print("Error: Could not load images.")
      return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors in both images
    template_keypoints, template_descriptors = sift.detectAndCompute(template_image, None)
    image_keypoints, image_descriptors = sift.detectAndCompute(target_image, None)

    if template_descriptors is not None and image_descriptors is not None:
        # Use FLANN-based matcher for feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        
        matches = flann.knnMatch(template_descriptors, image_descriptors, k=2)

        # Apply Lowe's ratio test to filter good matches
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        
        print(len(good_matches))
        if len(good_matches) > 10:  # Homography needs at least 4 points

          src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2) 
          dst_pts = np.float32([image_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

          # Find the homography matrix
          H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

          if H is not None:
            print("Found Homography")
            # Get the height and width of the template image
            h, w = template_image.shape

            # Define the corners of the template image
            template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

            # Use the homography to project the corners of the template onto the video frame
            projected_corners = cv2.perspectiveTransform(template_corners, H)

            # Draw the projected corners on the frame
            matched_frame = cv2.polylines(frame.copy(), [np.int32(projected_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
            perspective_fit_clue_board = cv2.warpPerspective(frame, np.linalg.inv(H), (w+400, h+220)) #400, 200 for clueboard1
            # Draw matches on the frame
            #matched_frame = cv2.drawMatches(template_image, template_keypoints, frame, image_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            return perspective_fit_clue_board

    # If no good matches or descriptors are found, return the original frame
    return None

def drive_forward():
    # Initialize the ROS node
    rospy.init_node('drive_forward', anonymous=True)

    # Create a publisher for the /B1/cmd_vel topic
    velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
    timer_publisher = rospy.Publisher('/score_tracker', String, queue_size=10)

    #Make sure you connect
    rospy.sleep(1)

    start_timer_msg = String()
    start_timer_msg.data = "SkibidiToilet,MeowMeow,0,NA" 
    rospy.loginfo("Starting the timer")
    timer_publisher.publish(start_timer_msg)

    # Set the loop rate (in Hz)
    rate = rospy.Rate(10)

    # Define the Twist message
    vel_msg = Twist()
    vel_msg.linear.x = 5.5  # Move forward at 0.5 m/s
    vel_msg.angular.z = -2.0  # Yes rotation

    rospy.loginfo("Car is moving")

    start_time = rospy.Time.now()
    while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < 5.0:
        velocity_publisher.publish(vel_msg)
        rate.sleep()

    # Stop the robot
    #vel_msg.linear.x = 0.0
    #vel_msg.angular.z = 0.0
    #velocity_publisher.publish(vel_msg)
    #rospy.loginfo("Car has stopped")

    # Stop the timer by publishing to /score_tracker
    #stop_timer_msg = String()
    #stop_timer_msg.data = "TeamName,password,-1,NA"  # Replace TeamName and password with actual values
    #rospy.loginfo("Stopping the timer")
    #timer_publisher.publish(stop_timer_msg)


if __name__ == '__main__':
    try:
        bridge = CvBridge()
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('controller_pkg')  
        template_path = f"{package_path}/resources/clue_template.png"
        rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, image_callback)
        drive_forward()
    except rospy.ROSInterruptException:
        pass

