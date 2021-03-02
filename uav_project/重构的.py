
import trimesh
import numpy as np
import cv2
np.set_printoptions(suppress=True)

def read_labelme_json(path):
    import json
    f = open(path, encoding='utf-8')
    text = f.read()
    data = json.loads(text)
    shapes=np.asarray(data['shapes'][0]['points'])
    return shapes

# from math import *
# def calR(ω,ϕ,κ):
#     return np.asarray([cos(ϕ)*cos(κ),
#     -cos(ϕ)*sin(κ),
#     sin(ϕ),
#     cos(ω)*sin(κ)+sin(ω)*sin(ϕ)*cos(κ),
#     cos(ω)*cos(κ)-sin(ω)*sin(ϕ)*sin(κ),
#     -sin(ω)*cos(ϕ),
#     sin(ω)*sin(κ)-cos(ω)*sin(ϕ)*cos(κ),
#     sin(ω)*cos(κ)+cos(ω)*sin(ϕ)*sin(κ),
#     cos(ω)*cos(ϕ)]).reshape([3,3])


# 用labelme标记的json文件
path = r'D:\desktop\files\codes\PycharmProjects\project_test\DSC_0157.json'
shapes=read_labelme_json(path)
shapes=np.reshape(shapes,[-1,1,2])
# print(shapes)

# 从3_calibrated_camera_parameters.txt读到的这张影像的内参

# fileName imageWidth imageHeight
# camera matrix K [3x3]
# radial distortion [3x1]
# tangential distortion [2x1]
# camera position t [3x1]
# camera rotation R [3x3]
# camera model m = K [R|-Rt] X

# DSC_0157.JPG 7360 4912
# 7363.51850023951465118444 0 3686.33719996831450771424
# 0 7363.51850023951465118444 2439.04083412635009153746
# 0 0 1
# -0.09005921067656787182 0.10773972435740564180 -0.01257054987797603419
# -0.00029570539858373248 -0.00027422056492206067
# 114.12600004808672338186 378.81985574952466322429 -2.31159761431545440757
# -0.49632901575943477734 0.86794436424468834890 -0.01816834310427051929
# 0.86649900151927328196 0.49400145706385401034 -0.07170802455019638366
# -0.05326338781351704077 -0.05133362440616233424 -0.99726018196053234366

K=np.asarray([7363.51850023951465118444, 0, 3686.33719996831450771424,
0, 7363.51850023951465118444, 2439.04083412635009153746,
0, 0, 1]).reshape([3,3])

D=np.asarray([-0.09005921067656787182 ,0.10773972435740564180 ,-0.00029570539858373248 ,
              -0.00027422056492206067 ,-0.01257054987797603419])


# from pixel to camera coor
shapes2=cv2.undistortPoints(shapes.copy(),K,D)
shapes2=np.reshape(shapes2,[shapes.shape[0],2])

# shapes3=np.asarray([[0.001,0.001],[7359.0,4911.0]])
# shapes3=np.reshape(shapes3,[-1,1,2])
# shapes3=cv2.undistortPoints(shapes3,K,D)

f=np.asarray([1]*shapes.shape[0]).reshape([-1,1])

shapes2=np.concatenate([shapes2,f],axis=1)
# print(shapes2)

# shapes1=cv2.undistortPoints(shapes.copy(),K,D,None,None,K)
# shapes1=np.reshape(shapes1,[shapes.shape[0],2])
# shapes1=np.concatenate([shapes1,f],axis=1)
# shapes1=shapes1@np.linalg.inv(K).T

# 从3_calibrated_external_camera_parameters.txt读到的这张影像的外参
# imageName X Y Z Omega Phi Kappa
# DSC_0157.JPG 332081.126000 3488911.819856 4093.688402 -2.946680 3.053212 119.804001

xyz=shapes2.copy()

XsYsZs=np.asarray([332081.126000 ,3488911.819856, 4093.688402 ])
# OmegaPhiKappa=np.asarray([-2.946680 ,3.053212, 119.804001])
# OmegaPhiKappa=np.deg2rad(OmegaPhiKappa)

# R=calR(OmegaPhiKappa[0],OmegaPhiKappa[1],OmegaPhiKappa[2])
# R1=np.linalg.inv(R)

# 这个是内参的最后三行
R2=np.asarray([-0.49632901575943477734, 0.86794436424468834890 ,-0.01816834310427051929,
0.86649900151927328196, 0.49400145706385401034 ,-0.07170802455019638366,
-0.05326338781351704077 ,-0.05133362440616233424 ,-0.99726018196053234366]).reshape([3,3])

# -0.49632901575943477734 0.86794436424468834890 -0.01816834310427051929
# 0.86649900151927328196 0.49400145706385401034 -0.07170802455019638366
# -0.05326338781351704077 -0.05133362440616233424 -0.99726018196053234366

# print(R.T)
# print(R1.T)
# print(R2.T)

print('xyz:',xyz)

XwYwZw=xyz@R2
# XwYwZw=xyz@R1

# XwYwZw1=np.zeros(XwYwZw.shape)
# XwYwZw1[:,0]=XwYwZw[:,1].copy()
# XwYwZw1[:,1]=XwYwZw[:,0].copy()
# XwYwZw1[:,2]=XwYwZw[:,2].copy()
XwYwZw1=XwYwZw.copy()
XwYwZw1=XwYwZw1/XwYwZw1[:,2:]
# XwYwZw1[:,0]=-XwYwZw1[:,0]
print('XwYwZw1:',XwYwZw1)

# 三维模型，最好是简化过的，这样更快
tri=trimesh.load(r'D:\desktop\files\codes\PycharmProjects\project_test\pc\all_poisson_simp.obj')
ray1=trimesh.ray.ray_triangle.RayMeshIntersector(tri)

ray_origins=XsYsZs.copy().reshape([1,3])
ray_origins=np.tile(ray_origins,[XwYwZw.shape[0],1])

# delta=np.asarray([331704,3489621,2240])-XsYsZs
# delta=np.asarray([331967.06,3488607.75,1637.374])-XsYsZs
# delta=delta/delta[2]
# delta=delta.reshape([-1,3])*-1
# print('real delta:',delta)

# locations, index_ray, index_tri=ray1.intersects_location(np.asarray([332081.126000 ,3488911.819856, 4093.688402 ]).reshape([1,3]),
#                                                          delta)
# print(locations)

ray_directions=np.asarray([0,0,-3]).reshape([1,3])
locations, index_ray, index_tri=ray1.intersects_location(ray_origins, -XwYwZw1)
# print('locations:',locations)

index_ray=index_ray.reshape([-1,1])
index_loc=np.concatenate([index_ray,locations],axis=1)
print(index_loc)

np.savetxt(r'D:\desktop\files\codes\PycharmProjects\project_test\pc\intersect.xyz',locations)