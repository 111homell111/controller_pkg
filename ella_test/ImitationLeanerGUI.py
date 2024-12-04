#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from python_qt_binding import loadUi

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospkg
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
from tf.transformations import quaternion_from_euler
import numpy as np

import torch
import torch.nn as nn
from CNNModel import DriveCNN

import os
import string
from std_msgs.msg import String
from collections import defaultdict
import pickle
import gzip
import time


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



def move_robot(linear_x, angular_z, duration):
    """
    Move the robot at the specified linear speed and angular speed for a given duration.

    :param linear_x: Linear speed (positive for forward, negative for backward) in m/s.
    :param angular_z: Angular speed in rad/s (positive for counter-clockwise rotation, negative for clockwise).
    :param duration: Duration of the movement in seconds.
    """
    # Initialize ROS node
    # rospy.init_node('move_robot', anonymous=True)

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





class ImitationLearner(QtWidgets.QMainWindow):

	def __init__(self):
		super(ImitationLearner, self).__init__()
		loadUi("./ImitationLearner.ui", self)

		rospy.init_node("ClueGUI", anonymous=True)
		# rospy.init_node('clue_counter', anonymous=True)

		self.cmd_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
		rospack = rospkg.RosPack()
		package_path = rospack.get_path('controller_pkg') 

		self.crosswalk = False # True if crosswalk is encountered
		self.start_crosswalk_wait = None # time robbie started waiting at crosswalk
		self.ped_detected = False #True if robbie detects pedestrian
		self.start_ped_wait = None # time robbie first saw pedestrian
		self.past_crosswalk = False # are we past the crosswalk
		self.previous_frame = None 
		self.clue_count = 0
		self.roundabout = False
		self.truck_detected = False
		self.start_truck_wait = None
		self.start_truck_wait = None
		self.past_roundabout = False

		self.last_clue_time = time.time()

		self.bridge = CvBridge()
		rospy.sleep(1)
		rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback, queue_size=1)

		rospy.Subscriber("/clue_count", String, self.clue_count_callback, queue_size = 1)
		self.debug = True

		self.use_model = False
		model_path = os.path.join(package_path, 'ella_test')
		self.CNNModel = DriveCNN()
		model_file = os.path.join(model_path, 'drive_cnn.pth')
		if not os.path.exists(model_file):
			raise FileNotFoundError(f"Model file not found: {model_file}")
		self.CNNModel.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
		self.CNNModel.eval()

		self.use_model_checkbox.stateChanged.connect(self.toggle_model)

		self.steering_area.setStyleSheet("background-color: lightgray; border: 1px solid black;")
		self.steering_area.setMouseTracking(True)
		self.setMouseTracking(True)

		self.speed_up_button.clicked.connect(self.speed_up)
		self.speed_down_button.clicked.connect(self.speed_down)
		self.rotate_up_button.clicked.connect(self.rotate_up)
		self.rotate_down_button.clicked.connect(self.rotate_up)

		self.linear_sensitivity = 1.5  # Scale for linear velocity
		self.angular_sensitivity = 2.8  # Scale for angular velocity

		# Timer to publish commands at a constant rate
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.publish_command)
		self.timer.start(100)  # Publish at 10 Hz


		if self.debug:
			self.data_timer = QtCore.QTimer(self)
			self.data_timer.timeout.connect(self.record_data)
			self.data_timer.start(25)  

		self.linear_velocity = 0.0
		self.angular_velocity = 0.0

		self.is_recording = False
		self.start_recording_button.clicked.connect(self.start_recording)
		self.stop_recording_button.clicked.connect(self.stop_recording)

		self.current_image = None
		#data should contain a tuple containing an image and a tuple containing linevelo and angular velo
		self.data = []

	def toggle_model(self, state):
		if state == Qt.Checked:
			self.use_model = True
			self.scroll_box.append("Using Model")
		else: 
			self.use_model = False
			self.scroll_box.append("Manual Mode")

	def start_recording(self):
		self.data = []
		self.is_recording = True
		if not self.debug:
			self.scrollbox.append("WARNING, DATALOGGING IS OFF")
		else: 
			self.scroll_box.append("Starting Recording")

	def stop_recording(self):
		self.is_recording = False
		self.scroll_box.append("Stopped Recording")
		self.linear_velocity = 0
		self.angular_velocity = 0

		options = QtWidgets.QFileDialog.Options()
		file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
			self,
			"Save List as Gzip Compressed Pickle File",
			"",
			"Gzip Pickle Files (*.pkl.gz);;All Files (*)",
			options=options
		)

		if file_path:
			# Ensure the file path ends with '.pkl.gz'
			if not file_path.endswith('.pkl.gz'):
				file_path += '.pkl.gz'

			# Save the list as a gzip compressed pickle file
			with gzip.open(file_path, 'wb') as file:
				pickle.dump(self.data, file)
			print(f"List saved to {file_path}")


	def speed_up(self):
		self.linear_sensitivity += 0.2
	def speed_down(self):
		self.linear_sensitivity -= 0.2
	def rotate_up(self):
		self.angular_sensitivity += 0.2
	def rotate_down(self):
		self.angular_sensitivity -= 0.2


	def mouseMoveEvent(self, event):
		# Get the position of the mouse relative to the steering area
		mouse_pos = event.pos()
		print(mouse_pos)
		widget_rect = self.steering_area.geometry()
		center_x = widget_rect.center().x()
		center_y = widget_rect.center().y()

		# Calculate offsets from the center
		offset_x = mouse_pos.x() - center_x
		offset_y = center_y - mouse_pos.y()  # Invert Y-axis for intuitive controls

		# Normalize offsets to a range [-1, 1] based on widget dimensions
		norm_x = offset_x / (widget_rect.width() / 2)
		norm_y = offset_y / (widget_rect.height() / 2)

		# Clamp normalized values to [-1, 1]
		norm_x = max(min(norm_x, 1.0), -1.0)
		norm_y = max(min(norm_y, 1.0), -1.0)

		# Apply exponential curve to normalized values
		expo_factor = 1.1  # Higher values make the control more exponential
		expo_x = (abs(norm_x)) ** expo_factor * norm_x
		expo_y = (abs(norm_y)) ** expo_factor * norm_y

		# Map exponential offsets to linear and angular velocities
		self.linear_velocity = self.linear_sensitivity * expo_y
		self.angular_velocity = -self.angular_sensitivity * expo_x # Reverse for intuitive steering

		# Optional: Limit the velocities to a maximum range
		self.linear_velocity = round(max(min(self.linear_velocity, 2.0), -2.0),1)
		self.angular_velocity = round(max(min(self.angular_velocity, 3.0), -4.0),1)

		self.scroll_box.append(f"{self.linear_velocity},{self.angular_velocity}")

	def publish_command(self):
		# Create and publish the Twist message
		cmd_msg = Twist()
		cmd_msg.linear.x = self.linear_velocity
		cmd_msg.angular.z = self.angular_velocity
		self.cmd_pub.publish(cmd_msg)

	def record_data(self):
		if self.current_image is not None and (self.linear_velocity != 0 or self.angular_velocity !=0):
			self.data.append((self.current_image, self.linear_velocity, self.angular_velocity))
			self.update_image_label(self.data_label, self.current_image)

	def clue_count_callback(self, msg):
		try:
			self.clue_count = int(msg.data)
			rospy.loginfo(f"Received clue count: {self.clue_count}")
			self.last_clue_time = time.time()
		except ValueError:
			rospy.logwarn(f"Invalid clue count received: {msg.data}. Unable to convert to integer.")


	def detect_crosswalk(self, frame):
		"""
		Detect a red line in the lower part of the frame.
		"""
		# Convert the frame to HSV color space
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		# define the red color range (Hue 165-180)
		lower_red1 = np.array([0, 50, 50])
		upper_red2 = np.array([6, 255, 255])
		mask1 = cv2.inRange(hsv, lower_red1, upper_red2)

		lower_red2 = np.array([165, 50, 50])
		upper_red2 = np.array([180, 255, 255])
		mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
		mask = cv2.bitwise_or(mask1, mask2)

		# only take bottom portion of the mask
		height, _ = mask.shape
		mask[:height // 3 * 2, :] = 0  # Keep only the bottom third

		# Find contours in the mask
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Check if any significant contour is detected
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > 500:  # Adjust the threshold based on the expected line size
				return True

		return False
	

	def detect_magenta(self, frame):
		"""
		Detect a red line in the lower part of the frame.
		"""
		# Convert the frame to HSV color space
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		# define the red color range (Hue 165-180)
		lower_magenta = np.array([145, 100, 100])
		upper_magenta = np.array([165, 255, 255])
		mask = cv2.inRange(hsv, lower_magenta, upper_magenta)

		# only take bottom portion of the mask
		height, _ = mask.shape
		mask[:height // 3 * 2, :] = 0  # Keep only the bottom third

		# Find contours in the mask
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Check if any significant contour is detected
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > 500:  # Adjust the threshold based on the expected line size
				return True

		return False
	
	def detect_traffic(self, current_frame, previous_frame, object, threshold=80):
		"""
		movement mask between the current and previous frames.
		
		Args:
			current_frame (numpy.ndarray): current frame in BGR format.
			previous_frame (numpy.ndarray): previous frame in BGR format.
			threshold (int): The minimum difference value to consider movement.
			
		Returns:
			movement_mask (numpy.ndarray): A binary mask highlighting areas of movement.
			largest_contour (list): The largest contour of the moving region, if any.
		"""
		
		if current_frame is None or previous_frame is None:
			print("One of the frames is None, skipping pedestrian detection.")
			return False
		
		traffic = False

		height, width = current_frame.shape[:2]
		if object == "pedestrian":
			previous_frame = previous_frame[(width // 3): (2*width // 3) ]
			current_frame = current_frame[(width // 3): (2*width // 3) ]
		elif object == "truck":
			previous_frame = previous_frame[0: (2*width // 4) ]
			current_frame = current_frame[0: (2*width // 4) ]


		# Convert frames to grayscale
		gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
		gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
		

		# Calculate the absolute difference
		diff = cv2.absdiff(gray_current, gray_previous)

		# Threshold the difference to get binary movement areas
		_, movement_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

		# Find contours of the movement mask
		contours, _ = cv2.findContours(movement_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if contours:
			# Find the largest contour by area
			largest_contour = max(contours, key=cv2.contourArea)
			traffic = True
	

		return traffic



	def image_callback(self, msg):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
			cv_image = cv2.resize(cv_image, (200, 200))[50:] #image needs to be 200w x 150h
			self.update_image_label(self.camera_label, cv_image)
			# print(f"{type(cv_image)}")
			
			self.current_image = cv_image

			if self.detect_crosswalk(cv_image) and not self.past_crosswalk:
				if not self.crosswalk:
					if self.start_crosswalk_wait is None:
						self.start_crosswalk_wait = time.time()
					self.use_model = False
					self.crosswalk = True
					# self.previous_frame = cv_image	

					# stop the robot
					self.linear_velocity = 0.0
					self.angular_velocity = 0.0
					self.publish_command()
					self.scroll_box.append("Stopping for pedestrian")
				
				# move forward if pedestrian doesn't show up for 10s
				elif time.time() - self.start_crosswalk_wait > 10 and self.crosswalk and not self.ped_detected:
					self.use_model = True
					self.scroll_box.append("No pedestrian, moving on.")
					self.past_crosswalk = True


			if self.crosswalk and not self.past_crosswalk:
				# mark when pedestrian starts crossing road
				if self.detect_traffic(self.current_image, self.previous_frame, "pedestrian", 80) and not self.ped_detected:
					self.start_ped_wait = time.time() 
					self.ped_detected = True
					self.use_model = False
					self.scroll_box.append("pedestrian detected")

				# move forward if 1s has passed since pedestrian first started crossing road
				elif self.detect_traffic(self.current_image, self.previous_frame, "pedestrian", 80) and self.ped_detected and time.time() - self.start_ped_wait > 2:
					self.scroll_box.append("pedestrian gone")
					self.use_model = True
					self.past_crosswalk = True
			
			if self.clue_count == 3 and not self.roundabout and not self.past_roundabout:
				self.use_model = False
				self.roundabout = True

				if self.start_truck_wait is None:
					self.start_truck_wait = time.time()

				# stop the robot
				self.linear_velocity = 0.0
				self.angular_velocity = 0.0
				self.publish_command()
				self.scroll_box.append("Stopping for truck")

			if self.roundabout and not self.past_roundabout:
				# mark when truck starts crossing
				if self.detect_traffic(self.current_image, self.previous_frame, "truck", 130) and not self.truck_detected:
					self.start_truck_wait = time.time() 
					self.truck_detected = True
					self.use_model = False
					self.scroll_box.append("truck detected")

				# move forward if 1s has passed since truck showed up
				elif self.detect_traffic(self.current_image, self.previous_frame, "truck", 130) and self.truck_detected and time.time() - self.start_truck_wait > 1.5:
					self.scroll_box.append("truck gone")
					self.use_model = True
					self.past_roundabout = True
						
			
					
			self.previous_frame = cv_image

			if time.time() - self.last_clue_time > 60:
				self.use_model = False
				self.linear_velocity = 0.0
				self.angular_velocity = 0.0
				self.publish_command()
				self.scroll_box.append("just fookin teleport")
				spawnTo_clue(3)			
				self.last_clue_time = time.time()


			if self.use_model:
				with torch.no_grad():
					image_tensor = torch.from_numpy(cv_image).unsqueeze(0).float()
					linear_pred, angular_pred = self.CNNModel(image_tensor)
					self.linear_velocity = linear_pred * 0.7
					self.angular_velocity = angular_pred 
				self.publish_command()

		except CvBridgeError as e:
			rospy.logwarn(f"Error converting ROS Image to OpenCV: {e}")


	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
							 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	#Takes opencv image
	def resize_frame(self, label, frame):
		label_width, label_height = label.width(), label.height()
		frame_height, frame_width = frame.shape[:2]

		# Scale factor while maintaining aspect ratio
		scale = min(label_width / frame_width, label_height / frame_height)
		new_width, new_height = int(frame_width * scale), int(frame_height * scale)

		return cv2.resize(frame, (new_width, new_height))

	#Takes opencv frame
	def update_image_label(self, label, frame):
		resized_frame = self.resize_frame(label, frame)
		label.setPixmap(self.convert_cv_to_pixmap(resized_frame))


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = ImitationLearner()
	myApp.show()
	sys.exit(app.exec_())