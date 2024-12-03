import cv2
import numpy as np
import rospkg
import rospy

class ImageProcessor():
	def __init__(self):
		
		#import the clue_template
		rospack = rospkg.RosPack()
		package_path = rospack.get_path('controller_pkg') 
		template_image_path = f"{package_path}/clue_reader/assets/clue_template.png" 
		self.template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
		if self.template_image is None:
			rospy.logerr("Failed to load template image.")
		else:
			rospy.loginfo(f"Loaded template image from {template_image_path}")
		

	def set_template_image(self, template_image):
		self.template_image = template_image

	def get_template_image(self):
		return self.template_image.copy()

	def sift_and_shift(self, raw_frame):
		"""
		@brief Takes the fizz clue rule symbol and performs sift to get a homography. Then does an inverse perspective transform and crops 
				the image to isolate the clue board.
				Uncomment matched frame if you want to see the SIFT feature lines

		@param frame: raw camera footage in OpenCV format
		@return perspective transformed and isolated/cropped image of the clue board
		"""

		if self.template_image is None or raw_frame is None:
			print("Error: could not load image in sift n' shift")

		target_image = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

		sift = cv2.SIFT_create()

		# Detect keypoints and descriptors in both images
		template_keypoints, template_descriptors = sift.detectAndCompute(self.template_image, None)
		image_keypoints, image_descriptors = sift.detectAndCompute(target_image, None)

		if template_descriptors is not None and image_descriptors is not None:
			index_params = dict(algorithm=0, trees=5)
			search_params = dict()
			flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
			
			matches = flann.knnMatch(template_descriptors, image_descriptors, k=2)

			# Apply Lowe's ratio test to filter good matches
			good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

			if len(good_matches) > 8:  # Homography needs at least 4 points
				src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2) 
				dst_pts = np.float32([image_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

				H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

				if H is not None:
					print("Found Homography")
					h, w = self.template_image.shape
					template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
					# Use the homography to project the corners of the template onto the video frame
					projected_corners = cv2.perspectiveTransform(template_corners, H)
					#matched_frame = cv2.polylines(raw_frame.copy(), [np.int32(projected_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

					#matched_frame = cv2.drawMatches(self.template_image, template_keypoints, raw_frame, image_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
					
					#THESE ARE MY SEXY HARDCODED CROPPING VALUES!
					perspective_fit_clue_board = cv2.warpPerspective(raw_frame, np.linalg.inv(H), (w+400, h+220)) #400, 200 for clueboard1
					return perspective_fit_clue_board[50:-50, 100:-90]

		# If no good matches or descriptors are found, return None
		return None

	def biggest_blue(self, raw_image):
		largest_area = 0
		masked_image = self.threshold_blue(raw_image, sl=110)
		contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		#cv2.drawContours(raw_image,[max(contours, key=cv2.contourArea)], -1, (255,0,0), 8)  # Draw with different colors
		
		if contours:
			sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
			largest_contour = max(contours, key=cv2.contourArea)
			largest_area = cv2.contourArea(largest_contour)
			print(f"Largest area: {largest_area}")
		return largest_area

	def order_corners(self, corners):
		"""
		Orders the corners in the correct sequence:
		Top-left, Top-right, Bottom-right, Bottom-left.
		"""
		# Sum of x and y will be smallest for top-left, largest for bottom-right
		# Difference (y - x) will be smallest for top-right, largest for bottom-left
		s = corners.sum(axis=1)
		diff = np.diff(corners, axis=1)

		# Top-left: smallest sum
		top_left = corners[np.argmin(s)]
		# Bottom-right: largest sum
		bottom_right = corners[np.argmax(s)]
		# Top-right: smallest difference
		top_right = corners[np.argmin(diff)]
		# Bottom-left: largest difference
		bottom_left = corners[np.argmax(diff)]

		return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


	def rect_and_detect(self, raw_frame):
		"""
		Perform inverse perspective transform on the largest blue rectangle in the image.
		"""
		blue_mask = self.threshold_blue(raw_frame, hl=0, hh=10, sl=0, sh=10, vl=90, vh=220) #this is no longer blue but grey, refactor later

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

		# Apply morphological closing
		blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

		# Step 2: Find contours
		contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if not contours:
			print("No contours found!")
			return None

		# Step 3: Find the largest contour
		largest_contour = max(contours, key=cv2.contourArea)

		# Step 4: Approximate the contour to 4 points
		epsilon = 0.02 * cv2.arcLength(largest_contour, True)
		approx = cv2.approxPolyDP(largest_contour, epsilon, True)

		if len(approx) != 4:
			print("The detected shape does not have 4 corners.")
			return None

		# Step 5: Extract and order corners
		corners = np.array([point[0] for point in approx], dtype="float32")
		ordered_corners = self.order_corners(corners)

		# Step 6: Define the destination points
		width, height = 550, 350
		destination_points = np.array([
			[0, 0],
			[width - 1, 0],
			[width - 1, height - 1],
			[0, height - 1]
		], dtype="float32")

		# Step 7: Compute the perspective transform matrix
		transform_matrix = cv2.getPerspectiveTransform(ordered_corners, destination_points)

		# Step 8: Apply the inverse perspective transform
		warped_image = cv2.warpPerspective(raw_frame, transform_matrix, (width, height))

		return warped_image


	def threshold_blue(self, image, hl=100, hh = 130, sl=80, sh = 255, vl=50, vh=255):
		"""
		please give me a BGR image
		"""
		hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
		lower_blue = np.array([hl, sl, vl])
		upper_blue = np.array([hh, sh, vh])
		blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

		#blue_only = cv2.bitwise_and(image, image, mask=blue_mask)
		return blue_mask

	
	
	
	def normalize_image_scale(self, image, desired_height):
		original_height, original_width = image.shape[:2]

		aspect_ratio = original_width / original_height
		new_width = int(desired_height * aspect_ratio)

		resized_image = cv2.resize(image, (new_width, desired_height))

		return resized_image
	
	def get_subimages(self, image):
		#Lets try to normalize here?
		image = self.normalize_image_scale(image, desired_height=400)
		h,w,_ = image.shape
		padx = 8
		pady = 5
		subimages = [[],[]]

		top = image[:int(h/3)]
		bottom = image[-int(h/2):]

		input_images = [top, bottom]

		for i, input_image in enumerate(input_images):
			blue_mask = self.threshold_blue(input_image)
			kernel = np.ones((2, 3), np.uint8)
			blue_mask = cv2.erode(blue_mask, kernel, iterations=1)

			contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
			contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
			for contour in contours:
				x, y, w, h = cv2.boundingRect(contour)
				#print(f"Area is {w*h}")
				if w * h > 1500:
					sub_image = input_image[y-pady:y + h+pady, x-padx:x + w+padx]
					sub_image = cv2.resize(sub_image, (25, 35))
					sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
					subimages[i].append(sub_image)

		return subimages





