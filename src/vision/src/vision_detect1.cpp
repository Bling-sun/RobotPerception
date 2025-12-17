#include "vision_detect.h"
#include <thread>
#include <fstream>
#include <algorithm>
#include <cuda_runtime_api.h>

using namespace std;

// -------- TensorRT Logger --------
class TrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      ROS_WARN("TensorRT: %s", msg);
    else
      ROS_INFO("TensorRT: %s", msg);
  }
};
static TrtLogger gLogger;
// --------------------------------

Yolo11Node::Yolo11Node(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : nh_(nh), pnh_(pnh), it_(nh_), shutting_down_(false)
{
  pnh_.param<std::string>("model_path", model_path_, "/home/ii/Documents/RLTP/src/vision/model/best.engine");
  pnh_.param<float>("conf_thresh", conf_thresh_, 0.86f);
  pnh_.param<float>("nms_iou_thresh", nms_iou_thresh_, 0.45f);
  pnh_.param<int>("input_w", input_w_, 640);
  pnh_.param<int>("input_h", input_h_, 640);
  pnh_.param<float>("fx", fx_, 615.0f);
  pnh_.param<float>("fy", fy_, 615.0f);
  pnh_.param<float>("cx", cx_, 320.0f);
  pnh_.param<float>("cy", cy_, 240.0f);

  color_sub_ = it_.subscribe("/camera/color/image_raw", 1, &Yolo11Node::colorCallback, this);
  depth_sub_ = it_.subscribe("/camera/depth/image_rect_raw", 1, &Yolo11Node::depthCallback, this);

  det_pub_ = nh_.advertise<vision_msgs::Detection2DArray>("vision_detector/detected_bounding_boxes", 5);
  vis_pub_ = nh_.advertise<sensor_msgs::Image>("vision_detector/detected_image", 5);
  cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("vision_detector/target_position", 5);

  if (!loadEngine(model_path_))
    throw std::runtime_error("Failed to load TensorRT engine.");

  std::thread(&Yolo11Node::inferenceLoop, this).detach();
}

Yolo11Node::~Yolo11Node()
{
  shutting_down_ = true;
  if (d_input_) cudaFree(d_input_);
  if (d_output_) cudaFree(d_output_);
  if (context_) context_->destroy();
  if (engine_) engine_->destroy();
  if (runtime_) runtime_->destroy();
}

void Yolo11Node::colorCallback(const sensor_msgs::ImageConstPtr& msg)
{
  std::lock_guard<std::mutex> lk(frame_mutex_);
  latest_color_ = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
}

void Yolo11Node::depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
  std::lock_guard<std::mutex> lk(frame_mutex_);
  if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1)
    latest_depth_ = cv_bridge::toCvShare(msg, "16UC1")->image.clone();
  else
    latest_depth_ = cv_bridge::toCvShare(msg, "32FC1")->image.clone();
}

bool Yolo11Node::loadEngine(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    ROS_ERROR("Engine not found: %s", path.c_str());
    return false;
  }
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> data(size);
  file.read(data.data(), size);
  file.close();

  runtime_ = nvinfer1::createInferRuntime(gLogger);
  engine_ = runtime_->deserializeCudaEngine(data.data(), size, nullptr);
  if (!engine_) return false;
  context_ = engine_->createExecutionContext();

  input_index_ = 0;
  output_index_ = engine_->getNbBindings() - 1;

  auto in_dims = engine_->getBindingDimensions(input_index_);
  auto out_dims = engine_->getBindingDimensions(output_index_);

  size_t in_elems = 1, out_elems = 1;
  for (int i = 0; i < in_dims.nbDims; ++i) in_elems *= in_dims.d[i];
  for (int i = 0; i < out_dims.nbDims; ++i) out_elems *= out_dims.d[i];

  input_size_bytes_ = in_elems * sizeof(float);
  output_size_bytes_ = out_elems * sizeof(float);
  cudaMalloc(&d_input_, input_size_bytes_);
  cudaMalloc(&d_output_, output_size_bytes_);
  return true;
}

std::vector<float> Yolo11Node::preprocess(const cv::Mat& img)
{
  cv::Mat resized, rgb;
  cv::resize(img, resized, cv::Size(input_w_, input_h_));
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
  std::vector<float> data(3 * input_h_ * input_w_);
  int idx = 0;
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i < input_h_; ++i)
      for (int j = 0; j < input_w_; ++j)
        data[idx++] = rgb.at<cv::Vec3f>(i, j)[c];
  return data;
}

float Yolo11Node::iou(const Detection& a, const Detection& b)
{
  int x1 = std::max(a.left, b.left);
  int y1 = std::max(a.top, b.top);
  int x2 = std::min(a.right, b.right);
  int y2 = std::min(a.bottom, b.bottom);
  int w = std::max(0, x2 - x1);
  int h = std::max(0, y2 - y1);
  float inter = w * h;
  float areaA = (a.right - a.left) * (a.bottom - a.top);
  float areaB = (b.right - b.left) * (b.bottom - b.top);
  return inter / (areaA + areaB - inter + 1e-6f);
}

std::vector<Detection> Yolo11Node::doNMS(const std::vector<Detection>& dets, float iou_thresh)
{
  std::vector<Detection> res;
  auto idx = std::vector<int>(dets.size());
  for (size_t i = 0; i < dets.size(); ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return dets[a].conf > dets[b].conf; });
  std::vector<bool> rem(dets.size(), false);
  for (size_t i = 0; i < idx.size(); ++i) {
    if (rem[idx[i]]) continue;
    res.push_back(dets[idx[i]]);
    for (size_t j = i + 1; j < idx.size(); ++j)
      if (!rem[idx[j]] && iou(dets[idx[i]], dets[idx[j]]) > iou_thresh)
        rem[idx[j]] = true;
  }
  return res;
}

float Yolo11Node::getDepthMeters(int u, int v)
{
  std::lock_guard<std::mutex> lk(frame_mutex_);
  if (latest_depth_.empty()) return NAN;
  if (u < 0 || v < 0 || u >= latest_depth_.cols || v >= latest_depth_.rows) return NAN;

  if (latest_depth_.type() == CV_16UC1) {
    uint16_t d = latest_depth_.at<uint16_t>(v, u);
    return d > 0 ? d * 0.001f : NAN;
  } else {
    float d = latest_depth_.at<float>(v, u);
    return (d > 0.01f && d < 30.0f) ? d : NAN;
  }
}

double Yolo11Node::getStableDepth(const cv::Mat &depth, const cv::Rect &bbox)
{

    int small_w = bbox.width / 5;
    int small_h = bbox.height / 5;
    int small_x = bbox.x + (bbox.width - small_w) / 2;
    int small_y = bbox.y + (bbox.height - small_h) / 2;

    cv::Rect small_box(small_x, small_y, small_w, small_h);
    small_box &= cv::Rect(0, 0, depth.cols, depth.rows); 

    std::vector<double> values;
    values.reserve(small_box.width * small_box.height);

 
    for(int y = small_box.y; y < small_box.y + small_box.height; ++y)
    {
        const uint16_t* row = depth.ptr<uint16_t>(y);
        for(int x = small_box.x; x < small_box.x + small_box.width; ++x)
        {
            uint16_t d = row[x];
            if(d > 200 && d < 8000) 
                values.push_back(static_cast<double>(d));
        }
    }

    if(values.empty())
        return -1.0; 

    std::sort(values.begin(), values.end());

    double q1 = values[values.size() * 1 / 4];
    double q3 = values[values.size() * 3 / 4];
    double iqr = q3 - q1;

    double low = q1 - 1.5 * iqr;
    double high = q3 + 1.5 * iqr;

    std::vector<double> filtered;
    filtered.reserve(values.size());
    for(double v : values)
    {
        if(v >= low && v <= high)
            filtered.push_back(v);
    }

    if(filtered.empty())
        return -1.0;

    double sum = 0.0;
    for(double v : filtered)
        sum += v;

    return sum / filtered.size();
}

bool Yolo11Node::depthValid(float d) { return std::isfinite(d) && d > 0.05f && d < 30.0f; }

geometry_msgs::PointStamped Yolo11Node::pixelToPoint3D(int u, int v, float depth, const ros::Time& stamp)
{
  geometry_msgs::PointStamped p;  
  p.header.stamp = stamp;
  p.header.frame_id = "camera_init";
  p.point.x = (u - cx_) * depth / fx_;
  p.point.y = (v - cy_) * depth / fy_;
  p.point.z = depth;
  return p;
}

std::vector<Detection> Yolo11Node::postprocess(const float* out,
                                               int n,
                                               int stride,
                                               int img_w,
                                               int img_h)
{
    std::vector<Detection> dets;
    auto out_dims = engine_->getBindingDimensions(output_index_);
    int C = out_dims.d[1];  
    int N = out_dims.d[2];  

    for (int i = 0; i < N; i++)
    {
        float cx   = out[0 * C * N + 0 * N + i];
        float cy   = out[0 * C * N + 1 * N + i];
        float ww   = out[0 * C * N + 2 * N + i];
        float hh   = out[0 * C * N + 3 * N + i];
        float conf = out[0 * C * N + 4 * N + i];

        if (conf < conf_thresh_) continue;
        int left   = (cx - ww / 2.0f) * img_w / input_w_;
        int top    = (cy - hh / 2.0f) * img_h / input_h_;
        int right  = (cx + ww / 2.0f) * img_w / input_w_;
        int bottom = (cy + hh / 2.0f) * img_h / input_h_;

        dets.push_back({left, top, right, bottom, conf});
    }
    return dets;
}

void Yolo11Node::publishDetections(const std::vector<Detection>& dets,
                                   const ros::Time& stamp,
                                   const cv::Mat& raw_img,
                                   double infer_ms,
                                   double depth_ms)
{
  vision_msgs::Detection2DArray arr;
  arr.header.stamp = stamp;
  arr.header.frame_id = "camera_init";

  sensor_msgs::PointCloud2 cloud;
  cloud.header = arr.header;
  cloud.height = 1;
  cloud.width = dets.size();
  cloud.is_dense = false;
  cloud.is_bigendian = false;

  sensor_msgs::PointCloud2Modifier mod(cloud);
  mod.setPointCloud2FieldsByString(1, "xyz");
  mod.resize(dets.size());

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");


  cv::Mat vis = raw_img.clone();

  for (const auto& d : dets) {
    if (d.conf < conf_thresh_) continue;

    vision_msgs::Detection2D det;
    vision_msgs::BoundingBox2D bb;

    bb.size_x = d.right - d.left;
    bb.size_y = d.bottom - d.top;
    bb.center.x = d.left + bb.size_x / 2.0;
    bb.center.y = d.top + bb.size_y / 2.0;

    det.bbox = bb;

    vision_msgs::ObjectHypothesisWithPose hyp;
    hyp.id = 0;
    hyp.score = d.conf;
    det.results.push_back(hyp);
    arr.detections.push_back(det);

    int u = bb.center.x;
    int v = bb.center.y;

    // float depth = getDepthMeters(u, v);

    // if (!depthValid(depth)) {
    //   for (int dy = -3; dy <= 3 && !depthValid(depth); ++dy)
    //     for (int dx = -3; dx <= 3 && !depthValid(depth); ++dx)
    //       depth = getDepthMeters(u + dx, v + dy);
    // }
    cv::Mat depth_copy;
    {
        
      std::lock_guard<std::mutex> lk(frame_mutex_);
      if (!latest_depth_.empty())
        depth_copy = latest_depth_.clone();
    }

    float depth = NAN;
    if (!depth_copy.empty())
    {
      cv::Rect box(d.left, d.top, d.right - d.left, d.bottom - d.top);
      double depth_raw = getStableDepth(depth_copy, box);
      if (depth_raw > 0)
          depth = depth_raw * 0.001f;
    }

    if (depthValid(depth)) {
      *iter_x = (u - cx_) * depth / fx_;
      *iter_y = (v - cy_) * depth / fy_;
      *iter_z = depth;
    } else {
      *iter_x = *iter_y = *iter_z = NAN;
    }
    ++iter_x; ++iter_y; ++iter_z;

    cv::rectangle(vis, cv::Point(d.left, d.top), cv::Point(d.right, d.bottom), cv::Scalar(0,255,0), 2);
    char buf[64];
    sprintf(buf, "conf %.2f", d.conf);
    cv::putText(vis, buf, cv::Point(d.left, d.top - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,0), 2);

    cv::circle(vis, cv::Point(u, v), 3, cv::Scalar(0,0,255), -1);
  }


  det_pub_.publish(arr);
  cloud_pub_.publish(cloud);


  sensor_msgs::ImagePtr img_msg =
      cv_bridge::CvImage(arr.header, "bgr8", vis).toImageMsg();
  vis_pub_.publish(img_msg);

  double total_ms = infer_ms + depth_ms;
  double fps = 1000.0 / total_ms;

  // ROS_INFO("[YOLO11] det=%ld | infer=%.2f ms | depth=%.2f ms | total=%.2f ms | FPS=%.1f",
  //          dets.size(), infer_ms, depth_ms, total_ms, fps);
}

void Yolo11Node::inferenceLoop()
{
  ros::Rate r(20);
  while (ros::ok() && !shutting_down_) {

    cv::Mat img;
    {
      std::lock_guard<std::mutex> lk(frame_mutex_);
      if (latest_color_.empty()) { r.sleep(); continue; }
      img = latest_color_.clone();
    }

    auto t0 = ros::Time::now();

    auto input = preprocess(img);
    cudaMemcpy(d_input_, input.data(), input_size_bytes_, cudaMemcpyHostToDevice);

    auto t_infer0 = ros::Time::now();
    void* bindings[] = { d_input_, d_output_ };
    context_->executeV2(bindings);

    std::vector<float> output(output_size_bytes_ / sizeof(float));
    cudaMemcpy(output.data(), d_output_, output_size_bytes_, cudaMemcpyDeviceToHost);
    auto t_infer1 = ros::Time::now();

    auto dets = postprocess(output.data(), output.size(), 5, img.cols, img.rows);
    dets = doNMS(dets, nms_iou_thresh_);

    auto t_depth0 = ros::Time::now();
    auto stamp = ros::Time::now();

    publishDetections(dets, stamp, img, (t_infer1 - t_infer0).toSec() * 1000.0, (ros::Time::now() - t_depth0).toSec() * 1000.0);

    r.sleep();
  }
}