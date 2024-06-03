#
"""

pip install numpy
pip install transforms3d

眼在手上，求解手眼标定的变换矩阵：

T_base2end: 基座到末端，即末端在机械臂坐标系下的位姿
T_end2cam:  末端到相机，即相机在末端坐标系下的位姿
T_cam2target: 相机到标定板，即标定板在相机坐标系下的位姿

T_base2target1 = T_cam2target1 * T_end2cam * T_base2end1
T_base2target2 = T_cam2target2 * T_end2cam * T_base2end2
T_base2target3 = T_cam2target3 * T_end2cam * T_base2end3
...

求解 T_end2cam ？

"""

import math
import numpy as np
import transforms3d as tfs

def get_matrix_eular_radu(x, y, z, rx, ry, rz):
    rmat = tfs.euler.euler2mat(math.radians(rx), math.radians(ry), math.radians(rz))
    rmat = tfs.affines.compose(np.squeeze(np.asarray((x, y, z))), rmat, [1, 1, 1])
    return rmat
 
 
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
 
 
def rot2quat_minimal(m):
    quat = tfs.quaternions.mat2quat(m[0:3, 0:3])
    return quat[1:]
 
 
def quatMinimal2rot(q):
    p = np.dot(q.T, q)
    w = np.sqrt(np.subtract(1, p[0][0]))
    return tfs.quaternions.quat2mat([w, q[0], q[1], q[2]])
 
 
# hand = [1.1988093940033604, -0.42405585264804424, 0.18828251788562061, 151.3390418721659, -18.612399542280507,
#         153.05074895025035,
#         1.1684831621733476, -0.183273375514656, 0.12744868246620855, -161.57083804238462, 9.07159838346732,
#         89.1641128844487,
#         1.1508343174145468, -0.22694301453461405, 0.26625166858469146, 177.8815855486261, 0.8991159570568988,
#         77.67286224959672]
hand = [
 
    # -0.05448,-0.15018,0.06552,89.61059916,-2.119943842,-1.031324031,
    #     -0.10149,-0.23025,0.04023,96.7725716,6.187944187,5.328507495,
    #     -0.10114,-0.2207,0.04853,97.00175472,5.729577951,1.375098708    毫米单位
 
-54.48,	-150.18,	65.52,  89.61059916,    -2.119943842,   -1.031324031,
-101.49,-230.25,    40.23,  96.7725716,      6.187944187,    5.328507495,
-101.14,-220.7	,   48.53,  97.00175472,    5.729577951,    1.375098708
 
        ]
 
# camera = [-0.16249272227287292, -0.047310635447502136, 0.4077761471271515, -56.98037030812389, -6.16739631361851,
#           -115.84333735802369,
#           0.03955405578017235, -0.013497642241418362, 0.33975949883461, -100.87129330834215, -17.192685528625265,
#           -173.07354634882094,
#           -0.08517949283123016, 0.00957852229475975, 0.46546608209609985, -90.85270962096058, 0.9315977976503153,
#           175.2059707654342]
 
camera = [
 
    # -0.0794887,-0.0812433,0.0246,0.0008,0.0033,0.0182,
    #       -0.078034,-0.0879632,0.4881494,-0.1085,0.0925,-0.1569,
    #       -0.1086702,-0.0881681,0.4240367,-0.1052,0.1251,-0.1124,
-79.4887,	-81.2433,	24.6,        0.0008,     0.0033,     0.0182,
-78.034,	-87.9632,	488.1494,    -0.1085,    0.0925,     -0.1569,
-108.6702,	-88.1681,	424.0367,    -0.1052,    0.1251,     -0.1124,
 
 
          ]
 
Hgs, Hcs = [], []
for i in range(0, len(hand), 6):
    Hgs.append(get_matrix_eular_radu(hand[i], hand[i + 1], hand[i + 2], hand[i + 3], hand[i + 4], hand[i + 5],))
    Hcs.append(
        get_matrix_eular_radu(camera[i], camera[i + 1], camera[i + 2], camera[i + 3], camera[i + 4], camera[i + 5]))
 
Hgijs = []
Hcijs = []
A = []
B = []
size = 0
for i in range(len(Hgs)):
    for j in range(i + 1, len(Hgs)):
        size += 1
        Hgij = np.dot(np.linalg.inv(Hgs[j]), Hgs[i])
        Hgijs.append(Hgij)
        Pgij = np.dot(2, rot2quat_minimal(Hgij))
 
        Hcij = np.dot(Hcs[j], np.linalg.inv(Hcs[i]))
        Hcijs.append(Hcij)
        Pcij = np.dot(2, rot2quat_minimal(Hcij))
 
        A.append(skew(np.add(Pgij, Pcij)))
        B.append(np.subtract(Pcij, Pgij))
MA = np.asarray(A).reshape(size * 3, 3)
MB = np.asarray(B).reshape(size * 3, 1)
# inv是求矩阵A的逆矩阵，pinv是求矩阵A的伪逆矩阵， 当本身就可逆的时候，二者结果相同
# 当A不可逆的时候只能用pinv求伪逆矩阵，照样能得到结果，但是存在一定的误差，不能忽略
Pcg_ = np.dot(np.linalg.pinv(MA), MB)
pcg_norm = np.dot(np.conjugate(Pcg_).T, Pcg_)
Pcg = np.sqrt(np.add(1, np.dot(Pcg_.T, Pcg_)))
Pcg = np.dot(np.dot(2, Pcg_), np.linalg.inv(Pcg))
Rcg = quatMinimal2rot(np.divide(Pcg, 2)).reshape(3, 3)
 
A = []
B = []
id = 0
for i in range(len(Hgs)):
    for j in range(i + 1, len(Hgs)):
        Hgij = Hgijs[id]
        Hcij = Hcijs[id]
        A.append(np.subtract(Hgij[0:3, 0:3], np.eye(3, 3)))
        B.append(np.subtract(np.dot(Rcg, Hcij[0:3, 3:4]), Hgij[0:3, 3:4]))
        id += 1
 
MA = np.asarray(A).reshape(size * 3, 3)
MB = np.asarray(B).reshape(size * 3, 1)
Tcg = np.dot(np.linalg.pinv(MA), MB).reshape(3, )
print(tfs.affines.compose(Tcg, np.squeeze(Rcg), [1, 1, 1]))