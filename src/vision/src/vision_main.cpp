#include "vision_detect.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vision_detect_node");
  ros::NodeHandle nh, pnh("~");

  Yolo11Node node(nh, pnh);
  ros::spin();
  
  return 0;
}