import matplotlib.pyplot as plt
import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """

    
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def mag(v):
    return np.dot(unit_vector(v),v)

def unit_perpen(v):
    x = np.array([0.0,0.0,1.0]) 
    v += [0.0]
    return unit_vector(np.cross(x,v))[0:2]
    

x = [-0.389,-0.160]
y = [0.222,0.231]
plt.scatter(x,y)
l = np.array([-0.272,0.201])
k = np.array([x[0],y[0]])
j = np.array([x[1],y[1]])

plt.scatter(l[0],l[1])
plt.scatter((j+unit_perpen(k-j)*0.06)[0],(j+unit_perpen(k-j)*0.06)[1])


d = np.cross(k-j, l-j)/np.linalg.norm(k-j)
d=mag(d)
print(d)
print(unit_perpen(k-j))
if d < 0.6:
    if np.dot(unit_perpen(k-j),l-k) > 0: 
        print(1)
        leav = l - unit_perpen(k-j)*0.06
        leav2 = l- unit_perpen(k-j)*d
    else: 
        leav = l + unit_perpen(k-j)*0.06
        leav2 = l + unit_perpen(k-j)*d
    x += [leav[0],leav2[0]]
    
    y += [leav[1],leav2[1]]
    
plt.plot(x,y)
plt.show()