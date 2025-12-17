cmake_minimum_required(VERSION 3.10)
project(vision)
SET(CMAKE_BUILD_TYPE "Release")

find_package (OpenCV REQUIRED)
find_package (CUDA REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  sensor_msgs
  image_transport
  vision_msgs
  geometry_msgs
  std_msgs
)

# set(TENSORRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu")
# set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs cv_bridge image_transport vision_msgs geometry_msgs std_msgs
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)
  
add_executable(vision_detect_node
  src/vision_detect.cpp 
  src/vision_main.cpp
  )
target_link_libraries(vision_detect_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  nvinfer
  # cuda
  )

install(TARGETS vision_detect_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)