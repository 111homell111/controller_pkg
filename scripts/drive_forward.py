#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

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
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    velocity_publisher.publish(vel_msg)
    rospy.loginfo("Car has stopped")

    # Stop the timer by publishing to /score_tracker
    stop_timer_msg = String()
    stop_timer_msg.data = "TeamName,password,-1,NA"  # Replace TeamName and password with actual values
    rospy.loginfo("Stopping the timer")
    timer_publisher.publish(stop_timer_msg)

if __name__ == '__main__':
    try:
        drive_forward()
    except rospy.ROSInterruptException:
        pass