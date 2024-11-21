cmake_minimum_required(VERSION 3.0.2)
project(controller_pkg)

## Find catkin macros and libraries
## REQUIRED components are used to find dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  std_msgs
)

## Declare a catkin package
catkin_package()

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)


## Add any other necessary installation or build instructions here if needed