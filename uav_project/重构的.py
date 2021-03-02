import shapefile
import trimesh
import numpy as np
import cv2
import glob
import tqdm
import os
np.set_printoptions(suppress=True)

def read_labelme_json(path):
    import json
    f = open(path, encoding='utf-8')
    text = f.read()
    data = json.loads(text)
    shapes=[]
    for s in data['shapes']:
        shapes.append(np.asarray(s['points']))
    # shapes=np.asarray(data['shapes'][0]['points'])
    return shapes

class camera_param:
    name=""
    K=0
    D=0
    XsYsZs=0
    R=0

def find_cp(camera_params,name):
    for cp in camera_params:
        if(cp.name==name):
            return cp
    return 0

def save_shapefile(nparray_list,path):
    # w = shapefile.Writer(path)
    # w.autoBalance = 1

    w = shapefile.Writer(path,shapeType=shapefile.POLYGON)
    w.field('FIRST_FLD', 'C', '40')
    for i,n in enumerate(nparray_list):
        w.poly([n.tolist()])
        w.record('FIRST_FLD',str(i) )
    w.close()

if __name__=="__main__":
    # 先把pix4d的数据读入
    camera_params=[]
    with open(r"I:\20141215大岗山\all\1_initial\params\all_calibrated_camera_parameters.txt", "r") as f:  # 打开文件
        internal_camera_params = f.read().split('\n') # 读取文件
    with open(r"I:\20141215大岗山\all\1_initial\params\all_calibrated_external_camera_parameters.txt", "r") as f:  # 打开文件
        external_camera_params = f.read().split('\n')  # 读取文件

    for i in range(1,len(external_camera_params)):
        external_camera_param=external_camera_params[i].split(' ')
        if(external_camera_param==['']):
            break
        cp=camera_param()
        cp.name=external_camera_param[0][:-4]
        cp.XsYsZs=np.asarray(external_camera_param[1:4]).astype(np.float)

        internal_camera_param=internal_camera_params[(i-1)*10+8:(i-1)*10+18]
        if(internal_camera_param[0].split(' ')[0]!=external_camera_param[0]):
            print('错误')
            # break
        cp.K=internal_camera_param[1:4]
        for i in range(len(cp.K)):
            cp.K[i]=cp.K[i].split(' ')
        cp.K=np.asarray(cp.K).astype(np.float)

        radial_distortion=internal_camera_param[4].split(' ')
        tangential_distortion=internal_camera_param[5].split(' ')

        cp.D=np.asarray([radial_distortion[0],radial_distortion[1],tangential_distortion[0],tangential_distortion[1],radial_distortion[2]]).astype(np.float)

        cp.R=internal_camera_param[-3:]
        for i in range(len(cp.R)):
            cp.R[i]=cp.R[i].split(' ')
        cp.R=np.asarray(cp.R).astype(np.float)

        camera_params.append(cp)

    json_files=[]
    json_files.extend(glob.glob(r"I:\wurenji0203\data\*\*\*.json"))
    json_files.sort()
    tri = trimesh.load(r"I:\20141215大岗山\all\2_densification\point_cloud\Mesh.obj")

    huapo=[]
    nishiliu=[]
    bengta=[]
    elses=[]

    for i,j in enumerate(tqdm.tqdm(json_files)):
        shapess = read_labelme_json(j)
        name=os.path.split(j)[1][:8]
        cp=find_cp(camera_params,name)
        if(cp!=0):
            for shapes in shapess:
                shapes = np.reshape(shapes, [-1, 1, 2])

                shapes2 = cv2.undistortPoints(shapes.copy(), cp.K, cp.D)
                shapes2 = np.reshape(shapes2, [shapes.shape[0], 2])

                # shapes3=np.asarray([[0.001,0.001],[7359.0,4911.0]])
                # shapes3=np.reshape(shapes3,[-1,1,2])
                # shapes3=cv2.undistortPoints(shapes3,K,D)

                f = np.asarray([1] * shapes.shape[0]).reshape([-1, 1])

                shapes2 = np.concatenate([shapes2, f], axis=1)
                # print(shapes2)

                # shapes1=cv2.undistortPoints(shapes.copy(),K,D,None,None,K)
                # shapes1=np.reshape(shapes1,[shapes.shape[0],2])
                # shapes1=np.concatenate([shapes1,f],axis=1)
                # shapes1=shapes1@np.linalg.inv(K).T

                # 从3_calibrated_external_camera_parameters.txt读到的这张影像的外参
                # imageName X Y Z Omega Phi Kappa
                # DSC_0157.JPG 332081.126000 3488911.819856 4093.688402 -2.946680 3.053212 119.804001

                xyz = shapes2.copy()

                XsYsZs = cp.XsYsZs
                # OmegaPhiKappa=np.asarray([-2.946680 ,3.053212, 119.804001])
                # OmegaPhiKappa=np.deg2rad(OmegaPhiKappa)

                # R=calR(OmegaPhiKappa[0],OmegaPhiKappa[1],OmegaPhiKappa[2])
                # R1=np.linalg.inv(R)

                # 这个是内参的最后三行
                R2 = cp.R

                # -0.49632901575943477734 0.86794436424468834890 -0.01816834310427051929
                # 0.86649900151927328196 0.49400145706385401034 -0.07170802455019638366
                # -0.05326338781351704077 -0.05133362440616233424 -0.99726018196053234366

                # print(R.T)
                # print(R1.T)
                # print(R2.T)

                # print('xyz:', xyz)

                XwYwZw = xyz @ R2
                # XwYwZw=xyz@R1

                # XwYwZw1=np.zeros(XwYwZw.shape)
                # XwYwZw1[:,0]=XwYwZw[:,1].copy()
                # XwYwZw1[:,1]=XwYwZw[:,0].copy()
                # XwYwZw1[:,2]=XwYwZw[:,2].copy()
                XwYwZw1 = XwYwZw.copy()
                XwYwZw1 = XwYwZw1 / XwYwZw1[:, 2:]
                # XwYwZw1[:,0]=-XwYwZw1[:,0]
                # print('XwYwZw1:', XwYwZw1)

                # 三维模型，最好是简化过的，这样更快
                ray1 = trimesh.ray.ray_triangle.RayMeshIntersector(tri)

                ray_origins = XsYsZs.copy().reshape([1, 3])
                ray_origins = np.tile(ray_origins, [XwYwZw.shape[0], 1])

                # delta=np.asarray([331704,3489621,2240])-XsYsZs
                # delta=np.asarray([331967.06,3488607.75,1637.374])-XsYsZs
                # delta=delta/delta[2]
                # delta=delta.reshape([-1,3])*-1
                # print('real delta:',delta)

                # locations, index_ray, index_tri=ray1.intersects_location(np.asarray([332081.126000 ,3488911.819856, 4093.688402 ]).reshape([1,3]),
                #                                                          delta)
                # print(locations)

                ray_directions = np.asarray([0, 0, -3]).reshape([1, 3])
                locations, index_ray, index_tri = ray1.intersects_location(ray_origins, -XwYwZw1)
                # print('locations:',locations)

                index_ray = index_ray.reshape([-1, 1])
                index_loc = np.concatenate([index_ray, locations], axis=1)

                index_loc = index_loc[np.argsort(index_loc[:, 0])]
                # print(index_loc)
                locations=index_loc[:,1:3]
                if ('滑坡' in j):
                    huapo.append(locations)
                # 泥石流
                elif ('泥石流' in j):
                    nishiliu.append(locations)
                # 崩塌
                elif ('崩塌' in j):
                    bengta.append(locations)
                else:
                    elses.append(locations)
        # if(i>10):
        #     break

    save_shapefile(huapo,'huapo')
    save_shapefile(nishiliu,'nishiliu')
    save_shapefile(bengta,'bengta')
    save_shapefile(elses,'elses')
    print()
                # np.savetxt(r'D:\desktop\files\codes\PycharmProjects\project_test\pc\intersect.xyz', locations)
