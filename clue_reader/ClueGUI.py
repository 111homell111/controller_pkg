#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import rospy
from sensor_msgs.msg import Image
import rospkg

import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np

from sift_and_shift import Sift_and_Shifter

class ClueGUI(QtWidgets.QMainWindow):

	def __init__(self):
		super(ClueGUI, self).__init__()
		loadUi("./ClueGUI.ui", self)

		rospy.init_node("ClueGUI", anonymous=True)
		rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback)
		rospack = rospkg.RosPack()
		package_path = rospack.get_path('controller_pkg') 

		self.sift_and_shifter = Sift_and_Shifter()
		self.update_image_label(self.template_label, self.sift_and_shifter.get_template_image())

		self.bridge = CvBridge()
		rospy.sleep(1)

	# Converts the OpenCV frame to QPixmap for display
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

	def image_callback(self, msg):
		try:
			# Convert the ROS Image message to an OpenCV image
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

			self.update_image_label(self.camera_label, cv_image)

			clue_board = self.sift_and_shifter.sift_and_shift(cv_image)

			if clue_board is not None:
				self.update_image_label(self.clue_board_label, clue_board)
			
		except CvBridgeError as e:
			rospy.logwarn(f"Error converting ROS Image to OpenCV: {e}")

	

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = ClueGUI()
	myApp.show()
	sys.exit(app.exec_())