import numpy as np
import open3d as o3d

class FacialICPRefiner:
    """
    遵循文本 4.5(1) 和 4.1.4 的要求：骨性特征加权局部修正。
    利用“点到面 ICP”并配合 Huber 鲁棒核，以骨性高稳健区域（残差小的高置信点）为引导进行加权精确重验，
    将因为下眼睑、嘴颊等软组织位移所引发的深度误差平推扣除，为深度学习解算提供基于纯物理量的保底拦截。
    """
    def __init__(self, mesh, distance_threshold=0.03):
        self.distance_threshold = distance_threshold
        
        # 将 trimesh 对象转化为 open3d 的网格并采样，加速术中实时运行
        self.target_mesh = o3d.geometry.TriangleMesh()
        self.target_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float64, copy=True))
        self.target_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces, dtype=np.int32, copy=True))
        
        if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0:
            self.target_mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(mesh.vertex_normals, dtype=np.float64, copy=True))
        else:
            self.target_mesh.compute_vertex_normals()
            
        # 根据面部面积，采样获取数千个控制点作为骨本结构特征匹配源
        self.model_pcd = self.target_mesh.sample_points_poisson_disk(number_of_points=8000)
        self.model_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        )

    def refine(self, pose_matrix, depth_image, K):
        # 1. 提取当前物理相机捕捉的实际深度点云
        im_h, im_w = depth_image.shape
        o3d_depth = o3d.geometry.Image(depth_image.astype(np.float32))
        
        # 构造相机的真实内参约束
        intrinsic = o3d.camera.PinholeCameraIntrinsic(im_w, im_h, K[0,0], K[1,1], K[0,2], K[1,2])
        camera_pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth, intrinsic, depth_scale=1.0, depth_trunc=3.0
        )
        
        if len(camera_pcd.points) < 100:
            return pose_matrix
            
        # 为“点到面 ICP”计算实时点云表面法线
        camera_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )

        # 2. 设定 Robust Kernel 拦截异常形变点
        # 文本设定4.1.4：Huber Loss 能够在小残差的骨区施加 L2 强约束，对大形变的软组织(脸颊/嘴唇)施加 L1 弱反馈。
        loss = o3d.pipelines.registration.HuberLoss(k=0.01) # Huber阈值设为1cm误差内
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        
        # 3. 针对网络下发的初验位姿进行有界收敛重验
        try:
            reg_p2l = o3d.pipelines.registration.registration_icp(
                source=self.model_pcd, 
                target=camera_pcd, 
                max_correspondence_distance=self.distance_threshold, 
                init=pose_matrix, 
                estimation_method=p2l,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
            )
            refined_pose = reg_p2l.transformation
            return refined_pose
        except Exception as e:
            return pose_matrix
