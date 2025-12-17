#ifndef VISION_DETECT_H
#define VISION_DETECT_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/BoundingBox2D.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Header.h>

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <vector>
#include <memory>
#include <string>
#include <mutex>

struct Detection {
  int left, top, right, bottom;
  float conf;
  int class_id;
};

class Yolo11Node {
public:
  Yolo11Node(ros::NodeHandle& nh, ros::NodeHandle& pnh);
  ~Yolo11Node();

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber color_sub_;
  image_transport::Subscriber depth_sub_;
  ros::Publisher det_pub_;   // bounding boxes
  ros::Publisher vis_pub_;
  ros::Publisher cloud_pub_; // all 3D points
 
  cv::Mat latest_color_, latest_depth_;
  std::mutex frame_mutex_;

  float fx_, fy_, cx_, cy_;
  std::string model_path_;
  float conf_thresh_;
  float nms_iou_thresh_;
  int input_w_, input_h_;
  bool shutting_down_;

  // TensorRT
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  void* d_input_ = nullptr;
  void* d_output_ = nullptr;
  size_t input_size_bytes_ = 0;
  size_t output_size_bytes_ = 0;
  int input_index_ = -1;
  int output_index_ = -1;

  
  void colorCallback(const sensor_msgs::ImageConstPtr& msg);
  void depthCallback(const sensor_msgs::ImageConstPtr& msg);
  void inferenceLoop();
  cv::Scalar getColor(int cls);

  bool loadEngine(const std::string& path);
  std::vector<float> preprocess(const cv::Mat& img);
  std::vector<Detection> postprocess(const float* output, int out_elems, int out_cols, int img_w, int img_h);
  std::vector<Detection> doNMS(const std::vector<Detection>& dets, float iou_thresh);
  std::vector<std::string> class_names_ = {"Blue", "Dead", "Red"};

  float getDepthMeters(int u, int v);
  double getStableDepth(const cv::Mat &depth, const cv::Rect &bbox);
  bool depthValid(float d);
  geometry_msgs::PointStamped pixelToPoint3D(int u, int v, float depth_m, const ros::Time& stamp);
  float iou(const Detection& a, const Detection& b);

  void publishDetections(const std::vector<Detection>& dets,
                       const ros::Time& stamp,
                       const cv::Mat& raw_img,
                       double infer_ms,
                       double depth_ms);
};

#endif // VISION_DETECT_H