# #Code by GVV Sharma
# #December 7, 2019
# #released under GNU GPL
# #Drawing a triangle given 3 sides

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# image = mpimg.imread('exit-ramp.jpg')
# plt.imshow(image)
# plt.show()

import sys                                          #for path to external scripts
#sys.path.insert(0, '/home/user/txhome/storage/shared/gitlab/res2021/july/conics/codes/CoordGeo')        #path to my scripts
sys.path.insert(0, '/home/karthikeya/Desktop/Probability/7/codes/CoordGeo')        #path to my scripts
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from params import omat

#sys.path.insert(0, '/home/user/txhome/storage/shared/gitlab/res2021/july/conics/codes/CoordGeo')        #path to my scripts

#if using termux
import subprocess
import shlex
#end if


#Given that, 
A = np.array([[1],[-1]])
B = np.array([[-4],[6]])
C = np.array([[-3],[-5]])

#From A and B, we can say that AB and BC are
AB=A-B
BC=B-C

#Now we calculate the constant terms of the vector equations of the lines perpendicular to AB and BC
constant_AB=AB.T@C
constant_BC=BC.T@A
#Now the cooefficient terms of the vector equations will be 
P=np.block([[AB.T],[BC.T]])

#and the constant terms of the vector equations will be
R=np.block([[constant_AB],[constant_BC]])

#Now we use the below commmand to solve both the equations and find orthocenter 
H = np.linalg.solve(P,R) #Orthocenter 
#Now we calculate the LHS of the question
product = (A - H).T@(B - C)
print(product)


A = np.array([1,-1])
B = np.array([-4, 6])
C = np.array([-3,-5])
D = alt_foot(A,B,C)
E = alt_foot(B,A,C)
F = alt_foot(C,A,B)

H = line_intersect(norm_vec(B,E),E,norm_vec(C,F),F)

#Generating all lines
x_AB = line_gen(A,B)	
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD = line_gen(A,alt_foot(A,B,C))
x_AE = line_gen(A,alt_foot(B,A,C))
x_BE = line_gen(B,alt_foot(B,A,C))
x_CF = line_gen(C,alt_foot(C,A,B))
x_AF = line_gen(A,alt_foot(C,A,B))
x_CH = line_gen(C,H)
x_BH = line_gen(B,H)
x_AH = line_gen(A,H)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD_1$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE_1$')
plt.plot(x_AE[0,:],x_AE[1,:],linestyle = 'dashed',label='$AE_1$')
plt.plot(x_CF[0,:],x_CF[1,:],label='$CF_1$')
plt.plot(x_AF[0,:],x_AF[1,:],linestyle = 'dashed',label='$AF_1$')
plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')

#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
H = H.reshape(-1,1)
tri_coords = np.block([[A,B,C,D,E,F,H]])
#tri_coords = np.vstack((A,B,C,alt_foot(A,B,C),alt_foot(B,A,C),alt_foot(C,A,B),H)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D_1','E_1','F_1','H']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/karthikeya/Desktop/Probability/Assignment_3/figs/figure1.png')



