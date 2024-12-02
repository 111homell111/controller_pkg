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

from ImageProcessor import ImageProcessor
from CNNModel import CNNModelFast

import torch
import torch.nn as nn

import os
import string

class ClueGUI(QtWidgets.QMainWindow):

	def __init__(self):
		super(ClueGUI, self).__init__()
		loadUi("./ClueGUI.ui", self)

		rospy.init_node("ClueGUI", anonymous=True)
		rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback, queue_size=1)
		rospack = rospkg.RosPack()
		package_path = rospack.get_path('controller_pkg') 

		self.bridge = CvBridge()
		rospy.sleep(1)

		self.ImageProcessor = ImageProcessor()
		self.update_image_label(self.template_label, self.ImageProcessor.get_template_image())

		model_path = os.path.join(package_path, 'clue_reader')

		self.CNNModel = CNNModelFast()
		model_file = os.path.join(model_path, 'test_cnn_fast.pth')
		if not os.path.exists(model_file):
			raise FileNotFoundError(f"Model file not found: {model_file}")
		self.CNNModel.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
		self.label_map = list(string.ascii_uppercase + string.digits)

		#self.is_processing = False

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
			cv_image = cv_image[200:-100]
			self.update_image_label(self.camera_label, cv_image)

			masked_image = self.ImageProcessor.threshold_blue(cv_image, hl=0, hh=10, sl=0, sh=10, vl=80, vh=220) #need to rename, no longer bluemask
			self.update_image_label(self.blue_mask_label, masked_image)
			largest_area = self.ImageProcessor.biggest_blue(cv_image)
		
			top_output = ""
			bottom_output = ""

			if largest_area > 10000: #2700
				clue_board = self.ImageProcessor.rect_and_detect(cv_image)
				if clue_board is not None:
					self.update_image_label(self.clue_board_label, clue_board)
					subimages = self.ImageProcessor.get_subimages(clue_board)

					if (subimages[0] != None):
						top_stacked_image = self.stack_images_vertically(subimages[0])
						self.update_image_label(self.top_subimages_label, top_stacked_image)
						self.top_pred_label.setText(self.predict_word(subimages[0]))

					
					if (subimages[1] != None):
						bottom_stacked_image = self.stack_images_vertically(subimages[1])
						self.update_image_label(self.bottom_subimages_label, bottom_stacked_image)
						self.bottom_pred_label.setText(self.predict_word(subimages[1]))
					
			else:
				pass
			
		except CvBridgeError as e:
			rospy.logwarn(f"Error converting ROS Image to OpenCV: {e}")


	def predict_word(self,letter_images):
		predicted_word = ""
		if letter_images != None and len(letter_images) != 0:
			with torch.no_grad():
				batch = torch.stack([torch.from_numpy(img).float() for img in letter_images])
				outputs = self.CNNModel(batch)
				_, predicted_labels = torch.max(outputs, 1)
				predicted_word = ''.join([self.label_map[label.item()] for label in predicted_labels])
		return predicted_word

			

	def stack_images_vertically(self, image_list):
		# Ensure all images are the same width
		widths = [img.shape[1] for img in image_list]
		max_width = max(widths)

		# Resize images to the same width while maintaining aspect ratio
		resized_images = []
		for img in image_list:
			height, width = img.shape[:2]
			if width != 0:
				new_height = int(height * (max_width / width))
				resized_img = cv2.resize(img, (max_width, new_height))
				resized_images.append(resized_img)

		# Stack images vertically
		stacked_image = np.vstack(resized_images)
		return stacked_image
	

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = ClueGUI()
	myApp.show()
	sys.exit(app.exec_())