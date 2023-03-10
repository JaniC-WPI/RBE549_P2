import numpy as np

def cheirality_check(C_list, R_list, X_list):
    # initialize a list to store valid camera poses and their corresponding 3D points
    valid_poses = []
    valid_X = []

    # loop over all combinations of two cameras
    for i in range(4):
        for j in range(i+1, 4):
            # calculate the camera matrix for each pair of cameras
            P1 = np.dot(R_list[i], np.hstack((np.identity(3), -C_list[i].reshape(3, 1))))
            P2 = np.dot(R_list[j], np.hstack((np.identity(3), -C_list[j].reshape(3, 1))))

            # check cheirality condition for each triangulated point
            num_valid = 0
            X_valid = []
            for k in range(len(X_list)):
                X_hom = np.hstack((X_list[k], 1))
                x1_hom = np.dot(P1, X_hom)
                x2_hom = np.dot(P2, X_hom)

                if (X_hom[2] > 0) and (x1_hom[2] > 0) and (x2_hom[2] > 0):
                    num_valid += 1
                    X_valid.append(X_list[k])

            # if all points are in front of both cameras, save the camera pair and their corresponding 3D points
            if num_valid == len(X_list):
                valid_poses.append((i, j))
                valid_X.append(X_valid)

    # if there is only one valid pair of cameras, return the corresponding camera pose and the best 3D point estimate
    if len(valid_poses) == 1:
        i, j = valid_poses[0]
        C = (C_list[i] + C_list[j]) / 2
        R = np.dot(R_list[i], R_list[j].T)
        X = np.mean(valid_X[0], axis=0)
        return C, R, X
    else:
        return None
