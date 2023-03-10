import numpy as np

# Given the rotation matrix R and camera pose t, and the 3D points P1 and P2 in homogeneous coordinates,
# check if the points satisfy the Triangulation Cheirality Condition
def triangulate(K, R1, C1, R2, C2, x1, x2):
   
    # Compute the projection matrices P1 and P2
    P1 = np.dot(K, np.dot(R1, np.hstack((np.eye(3), -C1.reshape(3, 1)))))
    P2 = np.dot(K, (R2, np.hstack((np.eye(3), -C2.reshape(3, 1)))))

    # Construct the A matrix
    A = np.vstack((x1[0] * P1[2] - P1[0],
                   x1[1] * P1[2] - P1[1],
                   x2[0] * P2[2] - P2[0],
                   x2[1] * P2[2] - P2[1]))




    # Compute the SVD of the A matrix
    _, _, V = np.linalg.svd(A)

    # Extract the 3D point from the last column of V
    X = V[-1, :4]

    # Normalize the homogeneous coordinate
    X /= X[3]

    
    return X[:3]




    



