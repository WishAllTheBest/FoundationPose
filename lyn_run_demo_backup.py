'''
Docstring for lyn_run_demo
liu yaning
2026.01.27
运行FoundationPose位姿估计demo的主入口脚本。
'''

from estimater import *
from datareader import *
import argparse

# 测试主入口，env:pose
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  # /demo_data/mustard0/mesh/textured_simple.obj /FoundationPose_manual/mesh/pallet.obj
  # /FoundationPose_manual/head/mesh/head.stl.obj /FoundationPose_manual/head/mesh/head_fp_canonical.obj
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/FoundationPose_manual/geminiHead/mesh/new_head_fp_canonical.obj')
  # /demo_data/mustard0 /FoundationPose_manual
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/FoundationPose_manual/kinect_data_with_calibrated_K')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=3)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_kinect_data_with_calibrated_K_mesh')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  logging.info("mesh loaded")
  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  print("to_origin:", to_origin)
  print("extents:", extents)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  logging.info("scorer initialization done")
  refiner = PoseRefinePredictor()
  logging.info("refiner initialization done")
  glctx = dr.RasterizeCudaContext()
  logging.info("glctx initialization done")
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      
      # overlay face mesh edges projected into the image
      # transform mesh into camera frame using the same pose used for export
      m_viz = mesh.copy()
      m_viz.apply_transform(pose)   # now vertices are in camera coordinates
      verts = m_viz.vertices
      ones = np.ones((verts.shape[0],1), dtype=verts.dtype)
      verts_h = np.hstack([verts, ones])            # (N,4) (homogeneous not strictly needed)
      v_cam = verts                             # (N,3) already in camera frame
      zs = v_cam[:, 2]
      proj = (reader.K @ v_cam.T)                # (3,N)
      proj2 = (proj[:2] / (proj[2:3] + 1e-8)).T # (N,2)

      h, w = color.shape[:2]
      faces = m_viz.faces
      for f in faces:
        if (zs[f] <= 1e-6).any():
          continue
        pts = proj2[f].astype(int)
        if ((pts[:,0] < 0) | (pts[:,0] >= w) | (pts[:,1] < 0) | (pts[:,1] >= h)).all():
          continue
        cv2.line(vis, tuple(pts[0]), tuple(pts[1]), (255,170,0), 1, lineType=cv2.LINE_AA)
        cv2.line(vis, tuple(pts[1]), tuple(pts[2]), (255,170,0), 1, lineType=cv2.LINE_AA)
        cv2.line(vis, tuple(pts[2]), tuple(pts[0]), (255,170,0), 1, lineType=cv2.LINE_AA)
      
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=4, transparency=0, is_input_rgb=True)

      cv2.imshow('vis', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

