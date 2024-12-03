#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from python_qt_binding import loadUi

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospkg

import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np

import torch
import torch.nn as nn
from CNNModel import DriveCNN

import os
import string
from collections import defaultdict
import pickle

class ImitationLearner(QtWidgets.QMainWindow):

	def __init__(self):
		super(ImitationLearner, self).__init__()
		loadUi("./ImitationLearner.ui", self)

		rospy.init_node("ClueGUI", anonymous=True)
		self.cmd_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
		rospack = rospkg.RosPack()
		package_path = rospack.get_path('controller_pkg') 

		self.bridge = CvBridge()
		rospy.sleep(1)
		rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback, queue_size=1)

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
		self.angular_sensitivity = 2.5  # Scale for angular velocity

		# Timer to publish commands at a constant rate
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.publish_command)
		self.timer.start(100)  # Publish at 10 Hz

		self.linear_velocity = 0.0
		self.angular_velocity = 0.0

		self.is_recording = False
		self.start_recording_button.clicked.connect(self.start_recording)
		self.stop_recording_button.clicked.connect(self.stop_recording)

		self.current_image = ""
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
		self.scroll_box.append("Starting Recording")

	def stop_recording(self):
		self.is_recording = False
		self.scroll_box.append("Stopped Recording")

		options = QtWidgets.QFileDialog.Options()
		file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
			self,
			"Save List as Pickle File",
			"",
			"Pickle Files (*.pkl);;All Files (*)",
			options=options
		)

		if file_path:
			# Ensure the file path ends with '.pkl'
			if not file_path.endswith('.pkl'):
				file_path += '.pkl'

			# Save the list as a pickle file
			with open(file_path, 'wb') as file:
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

	def image_callback(self, msg):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
			cv_image = cv2.resize(cv_image, (200, 200))[50:] #image needs to be 200w x 150h
			self.update_image_label(self.camera_label, cv_image)
			
			self.current_image = cv_image
			if self.current_image is not None and self.linear_velocity != 0 and self.angular_velocity !=0:
				self.data.append((self.current_image, self.linear_velocity, self.angular_velocity))

			if self.use_model:
				with torch.no_grad():
					image_tensor = torch.from_numpy(cv_image).unsqueeze(0).float()
					linear_pred, angular_pred = self.CNNModel(image_tensor)
					self.linear_velocity = linear_pred
					self.angular_velocity = angular_pred
				self.publish_command

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