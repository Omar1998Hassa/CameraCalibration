# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

import numpy as np
from random import sample

# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    #[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                      M33         vn ]
    #
    # Then you can solve this using least squares with the 'np.linalg.lstsq' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD via np.linalg.svd, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 
    #
    ##################
    # Your code here #
    ##################
    aMatrix=[]
    bVector=[]
    for i in range(len(Points_2D)):
        aMatrix.append([Points_3D[i][0], Points_3D[i][1], Points_3D[i][2], 1, 0, 0, 0, 0, -1 * Points_2D[i][0] * Points_3D[i][0],-1 * Points_2D[i][0] * Points_3D[i][1], -1 * Points_2D[i][0] * Points_3D[i][2]])
        aMatrix.append([0, 0, 0, 0, Points_3D[i][0], Points_3D[i][1], Points_3D[i][2], 1, -1 * Points_2D[i][1] * Points_3D[i][0],-1 * Points_2D[i][1] * Points_3D[i][1], -1 * Points_2D[i][1] * Points_3D[i][2]])
        #aMatrix.append([Points_3D[i][0], Points_3D[i][1], Points_3D[i][2], 1, 0, 0, 0, 0, -1*Points_2D[i][0]*Points_3D[i][0],-1*Points_2D[i][0]*Points_3D[i][1],-1*Points_2D[i][0]*Points_3D[i][2], -1*Points_2D[i][0] ])
        #aMatrix.append([0, 0, 0, 0, Points_3D[i][0], Points_3D[i][1], Points_3D[i][2], 1, -1*Points_2D[i][1]*Points_3D[i][0],-1*Points_2D[i][1]*Points_3D[i][1],-1*Points_2D[i][1]*Points_3D[i][2], -1*Points_2D[i][1] ])
        bVector.append(Points_2D[i][0])
        bVector.append(Points_2D[i][1])
    M, _, _, _=np.linalg.lstsq(aMatrix, bVector)
    #u,s,v = np.linalg.svd(aMatrix)
    #M = np.append(M,1)
    M = np.array(
        [[ M[0],  M[1],  M[2], M[3]], [M[4],  M[5],  M[6], M[7]], [ M[8],  M[9],  M[10], 1]])
    #print(M)
    #print("space")
    #print(s)
    #print("space")
    #print(v)
    #print("space")
    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    # Your total residual should be less than 1.
    #print('Randomly setting matrix entries as a placeholder')
   # M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
    #              [0.6750, 0.3152, 0.1136, 0.0480],
    #              [0.1020, 0.1725, 0.7244, 0.9932]])

    return M

# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
    ##################
    # Your code here #
    ##################
    Q=np.array([[-1*M[0][0],-1*M[0][1],-1*M[0][2]],[-1*M[1][0],-1*M[1][1],-1*M[1][2]],[-1*M[2][0],-1*M[2][1],-1*M[2][2]]])
    M4 = np.array([M[0][3],M[1][3],M[2][3]])
    #print(Q.shape)
    #print(M4.shape)
    Center = np.matmul(np.linalg.inv(Q), M4)
    # Replace this with the correct code
    # In the visualization you will see that this camera location is clearly
    # incorrect, placing it in the center of the room where it would not see all
    # of the points.
    #print(Center)
    #Center = np.array([1,1,1])

    return Center

# Returns the camera center matrix for a given projection matrix
# 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
# 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# 'F_matrix' is 3x3 fundamental matrix
def estimate_fundamental_matrix(Points_a,Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    ##################
    # Your code here #
    ##################
    U=[]
    for i in range(len(Points_a)):
        U.append([Points_a[i][0]*Points_b[i][0], Points_a[i][1]*Points_b[i][0], Points_b[i][0], Points_a[i][0]*Points_a[i][1], Points_a[i][1]*Points_b[i][1], Points_b[i][1], Points_a[i][0], Points_a[i][1], 1])
    u, s, vh = np.linalg.svd(np.array(U),full_matrices=True)
    F=vh[-1,:]
    F= np.reshape(F,(3,3))

    u_f,s_f,vh_f =np.linalg.svd(F)
    s_f[-1]=0
    F_matrix = np.matmul(np.matmul(u_f,np.diag(s_f)),vh_f)
    #print(F_matrix)


    # This is an intentionally incorrect Fundamental matrix placeholder
    #F_matrix = np.array([[0,0,-.0004],[0,0,.0032],[0,-0.0044,.1034]])

    return F_matrix

def Normalized_estimate_fundamental_matrix(Points_a,Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    ##################
    # Your code here #
    ##################
    PointsAMean = np.mean(Points_a,axis=0)
  #  print("Mean")
   # print(PointsAMean)
    CenteredApoints=Points_a-PointsAMean
   # print(CenteredApoints)
    varPointsA = np.var(CenteredApoints,axis=0)
   # print(varPointsA)
    sA = varPointsA**(0.5)
    #print(sA)
    Ta = np.matmul([[1 / sA[0], 0, 0],[0, 1 / sA[1], 0],[0, 0, 1]],[[1, 0, -PointsAMean[0]],[0, 1, -PointsAMean[1]],[0, 0, 1]])
    xA =np.append(Points_a, np.ones([len(Points_a), 1], dtype=np.int32), axis=1)
    #print(xA)
    #print(xA.shape)
    #print(Ta.shape)
    normalizedA=np.matmul(Ta,np.transpose(xA))
    normalizedA=np.transpose(normalizedA)

    PointsBMean = np.mean(Points_b, axis=0)
    CenteredBpoints = Points_b - PointsBMean
    varPointsB = np.var(CenteredBpoints,axis=0)
    sB = varPointsB ** (0.5)
    Tb = np.matmul([[1 / sB[0], 0, 0], [0, 1 / sB[1], 0], [0, 0, 1]],
                   [[1, 0, -PointsBMean[0]], [0, 1, -PointsBMean[1]], [0, 0, 1]])
    xB = np.append(Points_b, np.ones([len(Points_b), 1], dtype=np.int32), axis=1)
    normalizedB = np.matmul(Tb, np.transpose(xB))
    normalizedB = np.transpose(normalizedB)

    U = []
    for i in range(normalizedA.shape[0]):
        U.append([normalizedB[i][0]*normalizedA[i][0], normalizedB[i][0]*normalizedA[i][1], normalizedB[i][0], normalizedB[i][1]*normalizedA[i][0], normalizedB[i][1]*normalizedA[i][1], normalizedB[i][1],  normalizedA[i][0], normalizedA[i][1], 1])



    u, s, vh = np.linalg.svd(np.array(U),full_matrices=True)
    F=vh[-1,:]
    F= np.reshape(F,(3,3))

    u_f,s_f,vh_f =np.linalg.svd(F)
    s_f[-1]=0
    F_matrix_Normlized = np.matmul(np.matmul(u_f,np.diag(s_f)),vh_f)
    F_origin_matrix = np.matmul(np.matmul(np.transpose(Tb),F_matrix_Normlized),Ta)
    F_origin_matrix = np.array(F_origin_matrix)


    # This is an intentionally incorrect Fundamental matrix placeholder
    #F_matrix = np.array([[0,0,-.0004],[0,0,.0032],[0,-0.0044,.1034]])

    return F_origin_matrix

# Takes h, w to handle boundary conditions
def apply_positional_noise(points, h, w, interval=3, ratio=0.2):
    """ 
    The goal of this function to randomly perturbe the percentage of points given 
    by ratio. This can be done by using numpy functions. Essentially, the given 
    ratio of points should have some number from [-interval, interval] added to
    the point. Make sure to account for the points not going over the image 
    boundary by using np.clip and the (h,w) of the image. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.clip

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] ( note that it is <x,y> )
            - desc: points for the image in an array
        h :: int 
            - desc: height of the image - for clipping the points between 0, h
        w :: int 
            - desc: width of the image - for clipping the points between 0, h
        interval :: int 
            - desc: this should be the range from which you decide how much to
            tweak each point. i.e if interval = 3, you should sample from [-3,3]
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will have some number from 
            [-interval, interval] added to the point. 
    """
    ##################
    # Your code here #
    ##################
    pointz=points
    ChangedPoints=[] # array of changed Points
    #print("length of points")
    #print(len(points))
    for i in range (round(ratio*len(points))):
        index = np.random.randint(0,len(points))
        while(index in ChangedPoints): # checking that no point is change more than one time
            index =np.random.randint(0,len(points))
        ChangedPoints.append(index)
        x=(interval*2)*np.random.rand()-interval # x dimension random number generation
        if (x>w):
            points[index][0]=w
        elif (x<0):
            points[index][0]=0
        else:
            points[index][0]=x


        y = (interval * 2)*np.random.rand()-interval # y dimension random number generation

        if (y>h):
            points[index][1]=h
        elif (y<0):
            points[index][1]=0
        else:
            points[index][1]=y
    #print(ChangedPoints)
    return points

# Apply noise to the matches. 
def apply_matching_noise(points, ratio=0.2):
    """

    The goal of this function to randomly shuffle the percentage of points given 
    by ratio. This can be done by using numpy functions. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.random.shuffle  

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] 
            - desc: points for the image in an array
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will be randomly shuffled.
    """
    ##################
    # Your code here #
    ##################
    pointz =points
    ShuffledPoints = []  # array of changed Points
    for i in range(round(ratio * len(points)*0.5)): # multiplied by 0.5 because in each step 2 points are shuffled
        index1 = np.random.randint(0, len(points))
        index2 =  np.random.randint(0, len(points))
        while (index1 in ShuffledPoints):  # checking that the points shuffled are not shuffled again
            index1 = np.random.randint(0, len(points))
        while (index2 in ShuffledPoints):
            index2 = np.random.randint(0, len(points))
        ShuffledPoints.append(index1)# adding index 1 to the list of shuffled points
        ShuffledPoints.append(index2)# adding index 2 to the list of shuffled points
        PointAtIndex1 = points[index1]# interchanging content between the two indicies chosen for suffling, np.shuffle was not used because it shuffles all the values while we want to shuffle just a ratio of the points
        points[index1]=points[index2]
        points[index2]=PointAtIndex1
    #print(ShuffledPoints)
    return points


# Find the best fundamental matrix using RANSAC on potentially matching
# points
# 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
# matching points from pic_a and pic_b. Each row is a correspondence (e.g.
# row 42 of matches_a is a point that corresponds to row 42 of matches_b.
# 'Best_Fmatrix' is the 3x3 fundamental matrix
# 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
# of 'matches_a' and 'matches_b') that are inliers with respect to
# Best_Fmatrix.
def ransac_fundamental_matrix(matches_a, matches_b):
    # For this section, use RANSAC to find the best fundamental matrix by
    # randomly sampling interest points. You would reuse
    # estimate_fundamental_matrix() from part 2 of this assignment.
    # If you are trying to produce an uncluttered visualization of epipolar
    # lines, you may want to return no more than 30 points for either left or
    # right images.
    ##################
    # Your code here #
    ##################
   # print("matches a")
   # print(matches_a)
    #print("matches b")
    #print(matches_b)
    tolerance =0.07
    previousError = 99999
    previousCount=0
    previousErrorList=[]
    for i in range(2000):
        subMatchesB=[]
        subMatchesA=[]
        indicies=[]
        errorList=[]
        currentError=0
        for i in range(8):
            index = np.random.randint(0, len(matches_a))
            while (index in indicies):  # checking that no point is change more than one time
                index = np.random.randint(0, len(matches_a))
            indicies.append(index)
            subMatchesA.append(matches_a[index])
            subMatchesB.append(matches_b[index])
       # print("match list")
        subMatchesB=np.array(subMatchesB)
        subMatchesA = np.array(subMatchesA)
        #index = np.random.randint(0, len(matches_a))
        CurrentFmatrix = Normalized_estimate_fundamental_matrix(subMatchesA,subMatchesB)
        CurrentInliersA=[]
        CurrentInliersB=[]
        CurrentCount=0
       # print("Fmatrix")
        #print(CurrentFmatrix)

        for j in range(len(matches_a)):
            #print(matches_a[j])
            ImageAChosenPoints = np.array([matches_a[j][0],matches_a[j][1], 1 ])
            ImageBChosenPoints=np.array([matches_b[j][0],matches_b[j][1], 1 ])
            #print(ImageBChosenPoints.shape)

            dist= np.matmul(np.matmul(np.expand_dims(ImageBChosenPoints,axis=0),CurrentFmatrix),np.transpose(np.expand_dims(ImageAChosenPoints,axis=0)))
           # print("Dist")
           # print(dist)
            if(abs(dist) < tolerance):
                errorList.append(dist)
                currentError = currentError + abs(dist)
                CurrentCount=CurrentCount+1
                CurrentInliersA.append(matches_a[j])
                CurrentInliersB.append(matches_b[j])

        #meanCurrentError =currentError/CurrentCount
        if(CurrentCount>=previousCount and currentError<previousError):
            previousErrorList = errorList
            previousError=currentError
            Best_Fmatrix = CurrentFmatrix
            previousCount = CurrentCount
            previousInliersA = np.array(CurrentInliersA)
            previousInliersB = np.array(CurrentInliersB)


    # Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.

    # placeholders, you can delete all of this
    #Best_Fmatrix = estimate_fundamental_matrix(matches_a[0:9,:],matches_b[0:9,:])
    #inliers_a = matches_a[0:29,:]
    #inliers_b = matches_b[0:29,:]
   # print("The Count: " + str(previousCount) +"")
    inliers_a=[]
    inliers_b=[]
    #for i in range(30):
        #index=previousErrorList.index(min(previousErrorList))
        #pop = previousErrorList.pop(index)
        #inliers_a.append(previousInliersA[index][:])
        #inliers_b.append(previousInliersB[index][:])

    inliers_a = np.array(previousInliersA)
    inliers_b = np.array(previousInliersB)
    return Best_Fmatrix, inliers_a, inliers_b
