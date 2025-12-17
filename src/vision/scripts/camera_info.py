import pyrealsense2 as rs
import numpy as np
import json

def get_d455_intrinsics():
    """获取D455相机的内参"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        # 启用彩色和深度流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 启动管道
        pipeline.start(config)
        
        # 等待几帧让相机稳定
        for i in range(10):
            pipeline.wait_for_frames()
        
        # 获取设备
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        
        # 获取内参
        color_intrinsics = color_profile.get_intrinsics()
        depth_intrinsics = depth_profile.get_intrinsics()
        
        print("=" * 50)
        print("D455 相机内参")
        print("=" * 50)
        
        # 打印彩色相机内参
        print("\n📷 彩色相机内参:")
        print(f"  宽度: {color_intrinsics.width}")
        print(f"  高度: {color_intrinsics.height}")
        print(f"  焦距: [{color_intrinsics.fx:.6f}, {color_intrinsics.fy:.6f}]")
        print(f"  光学中心: [{color_intrinsics.ppx:.6f}, {color_intrinsics.ppy:.6f}]")
        print(f"  畸变模型: {color_intrinsics.model.name}")
        print(f"  畸变系数: {color_intrinsics.coeffs}")
        
        # 打印深度相机内参
        print("\n📏 深度相机内参:")
        print(f"  宽度: {depth_intrinsics.width}")
        print(f"  高度: {depth_intrinsics.height}")
        print(f"  焦距: [{depth_intrinsics.fx:.6f}, {depth_intrinsics.fy:.6f}]")
        print(f"  光学中心: [{depth_intrinsics.ppx:.6f}, {depth_intrinsics.ppy:.6f}]")
        print(f"  畸变模型: {depth_intrinsics.model.name}")
        print(f"  畸变系数: {depth_intrinsics.coeffs}")
        
        # 保存为JSON文件
        intrinsics_data = {
            "color_camera": {
                "width": color_intrinsics.width,
                "height": color_intrinsics.height,
                "fx": color_intrinsics.fx,
                "fy": color_intrinsics.fy,
                "ppx": color_intrinsics.ppx,
                "ppy": color_intrinsics.ppy,
                "model": color_intrinsics.model.name,
                "coeffs": color_intrinsics.coeffs
            },
            "depth_camera": {
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "ppx": depth_intrinsics.ppx,
                "ppy": depth_intrinsics.ppy,
                "model": depth_intrinsics.model.name,
                "coeffs": depth_intrinsics.coeffs
            }
        }
        
        with open('d455_intrinsics.json', 'w') as f:
            json.dump(intrinsics_data, f, indent=4)
        
        print(f"\n💾 内参已保存到: d455_intrinsics.json")
        
        return color_intrinsics, depth_intrinsics
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None, None
        
    finally:
        pipeline.stop()

def load_intrinsics_from_file(filename='d455_intrinsics.json'):
    """从文件加载内参"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # 创建内参对象（简化版本）
        class Intrinsics:
            def __init__(self, data):
                self.width = data['width']
                self.height = data['height']
                self.fx = data['fx']
                self.fy = data['fy']
                self.ppx = data['ppx']
                self.ppy = data['ppy']
                self.model = type('Model', (), {'name': data['model']})()
                self.coeffs = data['coeffs']
        
        color_intrinsics = Intrinsics(data['color_camera'])
        depth_intrinsics = Intrinsics(data['depth_camera'])
        
        return color_intrinsics, depth_intrinsics
        
    except FileNotFoundError:
        print(f"❌ 文件 {filename} 未找到，请先运行 get_d455_intrinsics()")
        return None, None

def create_camera_matrix(intrinsics):
    """从内参创建相机矩阵"""
    return np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

def create_distortion_coeffs(intrinsics):
    """从内参创建畸变系数"""
    return np.array(intrinsics.coeffs)

# 使用示例
if __name__ == "__main__":
    # 方法1: 直接从相机读取内参
    color_intr, depth_intr = get_d455_intrinsics()
    
    if color_intr and depth_intr:
        # 创建相机矩阵
        color_matrix = create_camera_matrix(color_intr)
        depth_matrix = create_camera_matrix(depth_intr)
        
        color_distortion = create_distortion_coeffs(color_intr)
        depth_distortion = create_distortion_coeffs(depth_intr)
        
        print("\n🎯 彩色相机矩阵:")
        print(color_matrix)
        print("\n🎯 彩色畸变系数:")
        print(color_distortion)
        
        print("\n🎯 深度相机矩阵:")
        print(depth_matrix)
        print("\n🎯 深度畸变系数:")
        print(depth_distortion)
    
    # 方法2: 从文件加载内参
    print("\n" + "="*50)
    print("从文件加载内参")
    print("="*50)
    
    color_intr_loaded, depth_intr_loaded = load_intrinsics_from_file()
    
    if color_intr_loaded and depth_intr_loaded:
        print("✅ 内参加载成功!")
        print(f"彩色相机焦距: [{color_intr_loaded.fx:.2f}, {color_intr_loaded.fy:.2f}]")