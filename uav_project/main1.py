
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

from math import *
def calR(ω,ϕ,κ):
    return np.asarray([cos(ϕ)*cos(κ),
    -cos(ϕ)*sin(κ),
    sin(ϕ),
    cos(ω)*sin(κ)+sin(ω)*sin(ϕ)*cos(κ),
    cos(ω)*cos(κ)-sin(ω)*sin(ϕ)*sin(κ),
    -sin(ω)*cos(ϕ),
    sin(ω)*sin(κ)-cos(ω)*sin(ϕ)*cos(κ),
    sin(ω)*cos(κ)+cos(ω)*sin(ϕ)*sin(κ),
    cos(ω)*cos(ϕ)]).reshape([3,3])



path = r'/mnt/hgfs/E/project_test/DSC_0027.json'
shapes=read_labelme_json(path)
shapes=np.reshape(shapes,[-1,1,2])
# print(shapes)

K=np.asarray([7363.51850023951465118444, 0, 3686.33719996831450771424,
0, 7363.51850023951465118444, 2439.04083412635009153746,
0, 0, 1]).reshape([3,3])

D=np.asarray([-0.09005921067656787182 ,0.10773972435740564180 ,-0.00029570539858373248 ,
              -0.00027422056492206067 ,-0.01257054987797603419])

# shapes1=cv2.undistortPoints(shapes.copy(),K,D,None,None,K)
# shapes1=np.reshape(shapes1,[shapes.shape[0],2])

# from pixel to camera coor
shapes2=cv2.undistortPoints(shapes.copy(),K,D)
shapes2=np.reshape(shapes2,[shapes.shape[0],2])

# shapes3=np.asarray([[0.001,0.001],[7359.0,4911.0]])
# shapes3=np.reshape(shapes3,[-1,1,2])
# shapes3=cv2.undistortPoints(shapes3,K,D)

f=np.asarray([1]*shapes.shape[0]).reshape([-1,1])

shapes2=np.concatenate([shapes2,f],axis=1)
# print(shapes2)


xyz=shapes2.copy()
# 332081.126000 3488911.819856 4093.688402 -2.946680 3.053212 119.804001
XsYsZs=np.asarray([328000.153872 ,3486121.505272 ,4096.303396])
OmegaPhiKappa=np.asarray([1.499460 ,-3.689015 ,-59.902840])
OmegaPhiKappa=np.deg2rad(OmegaPhiKappa)

R=calR(OmegaPhiKappa[0],OmegaPhiKappa[1],OmegaPhiKappa[2])
R1=np.linalg.inv(R)
R2=np.asarray([0.50042880166716996460, -0.86572430855446047548, 0.00961436632656934541,
-0.86338360663329860589, -0.49983949117315368271, -0.06876940352104310428,
0.06434098428558245042 ,0.02611330392087448521 ,-0.99758625346357809871]).reshape([3,3])

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


tri=trimesh.load('/mnt/hgfs/E/project_test/pc/all_poisson_simp.obj')
ray1=trimesh.ray.ray_triangle.RayMeshIntersector(tri)

ray_origins=XsYsZs.copy().reshape([1,3])
ray_origins=np.tile(ray_origins,[XwYwZw.shape[0],1])

# delta=np.asarray([331704,3489621,2240])-XsYsZs
delta=np.asarray([327820.5,3486489,2399.559])-XsYsZs
delta=delta/delta[2]
delta=delta.reshape([-1,3])*-1
print('real delta:',delta)

# locations, index_ray, index_tri=ray1.intersects_location(np.asarray([332081.126000 ,3488911.819856, 4093.688402 ]).reshape([1,3]),
#                                                          delta)
# print(locations)

ray_directions=np.asarray([0,0,-3]).reshape([1,3])
locations, index_ray, index_tri=ray1.intersects_location(ray_origins, -XwYwZw1)
# print('locations:',locations)

index_ray=index_ray.reshape([-1,1])
index_loc=np.concatenate([index_ray,locations],axis=1)
print(index_loc)

np.savetxt('/mnt/hgfs/E/project_test/pc/intersect1.xyz',locations)