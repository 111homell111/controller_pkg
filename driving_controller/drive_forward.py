#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_euler
import cv2



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


# Callback for the camera image
def camera_callback(msg):
    rospy.loginfo("Received an image!")

# Callback for the clock
def clock_callback(msg):
    rospy.loginfo(f"Simulated time: {msg.clock.secs} seconds")



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




def drive_forward():
    # Initialize the ROS node
    rospy.init_node('drive_forward', anonymous=True)

    # Create a publisher for the /B1/cmd_vel topic
    velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
    timer_publisher = rospy.Publisher('/score_tracker', String, queue_size=10)

    #Make sure you connect
    rospy.sleep(1)

    # Create subscribers
    rospy.Subscriber('/B1/camera1/image_raw', Image, camera_callback)
    rospy.Subscriber('/clock', Clock, clock_callback)

    # Wait for the Gazebo service to become available
    rospy.wait_for_service('/gazebo/set_model_state')
    set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

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
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    velocity_publisher.publish(vel_msg)
    rospy.loginfo("Car has stopped")


    # Call the Gazebo service to set the model state
    state_msg = ModelState()
    state_msg.model_name = "B1"
    state_msg.pose.position.x = 0.0
    state_msg.pose.position.y = 0.0
    state_msg.pose.position.z = 0.0
    state_msg.pose.orientation.x = 0.0
    state_msg.pose.orientation.y = 0.0
    state_msg.pose.orientation.z = 0.0
    state_msg.pose.orientation.w = 1.0
    state_msg.twist.linear.x = 0.0
    state_msg.twist.linear.y = 0.0
    state_msg.twist.linear.z = 0.0
    state_msg.twist.angular.x = 0.0
    state_msg.twist.angular.y = 0.0
    state_msg.twist.angular.z = 0.0
    state_msg.reference_frame = "world"

    try:
        response = set_model_state(state_msg)
        if response.success:
            rospy.loginfo("Model state set successfully")
        else:
            rospy.logwarn("Failed to set model state")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


    # Stop the timer by publishing to /score_tracker
    stop_timer_msg = String()
    stop_timer_msg.data = "TeamName,password,-1,NA"  # Replace TeamName and password with actual values
    rospy.loginfo("Stopping the timer")
    timer_publisher.publish(stop_timer_msg)


def move_robot(linear_x, angular_z, duration):
    """
    Move the robot at the specified linear speed and angular speed for a given duration.

    :param linear_x: Linear speed (positive for forward, negative for backward) in m/s.
    :param angular_z: Angular speed in rad/s (positive for counter-clockwise rotation, negative for clockwise).
    :param duration: Duration of the movement in seconds.
    """
    # Initialize ROS node
    rospy.init_node('move_robot', anonymous=True)

    # Create a publisher for the /B1/cmd_vel topic
    velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)

    # pause for ensuring connections
    rospy.sleep(1)

    # set the loop rate (Hz)
    rate = rospy.Rate(10)

    # define twist message
    vel_msg = Twist()
    vel_msg.linear.x = linear_x  # linear speed in x-direction
    vel_msg.angular.z = angular_z  # angular speed in z-direction

    rospy.loginfo(f"Car is moving with linear speed {linear_x} m/s and angular speed {angular_z} rad/s")

    # Start moving
    start_time = rospy.Time.now()
    while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < duration:
        velocity_publisher.publish(vel_msg)
        rate.sleep()

    # Stop the robot after the duration
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    velocity_publisher.publish(vel_msg)
    rospy.loginfo("Car has stopped")


def spawnTo_clue(clue):
    if (clue == 1):
        spawn_position(FIRST_SPAWN)
        move_robot(-4,0,0.65)
        move_robot(0,-6,0.18)
        move_robot(2,0,0.3)
        move_robot(0,0,0.2)
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

if __name__ == '__main__':
    try:
        # move_robot(2, 0, 0.49)
        # move_robot(0,6, 0.18)
        # move_robot(2,0,0.3)
        # move_robot(0,0,0.2)
        # spawnTo_clue(1)
        spawnTo_clue(3)
    except rospy.ROSInterruptException:
        pass