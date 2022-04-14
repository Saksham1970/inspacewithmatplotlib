from tkinter import image_names
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
import pandas as pd
import math
import matplotlib.patches as patches
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont
#? Parameters definitions

# Image Name
image_name = "myimage.png"

# Node Images Length and Width
node_length = 0.4
node_width = 0.24

figure_size = (96,60)

min_dist_bw_two_nodes = 0.12
node_clearance = 2
min_edge_dist_from_node = 0.5
arrow_distance = 0.4

# Start and End Card Lengthe and Width
card_length = 0.6
card_width = 0.6

# Edges/Arrows/Paths colors list
# They are chosen in ascending order on intersecting lines, to better identify the lines
# Add more colors in the end of the list in case of more intersections
edge_color_list = ["#99AAB5", "#99b5a1", "#b599b5", "#b5ab99"]

# Colors for the rest of the graph
color_palette = {
    "background" : "#121212",
    "title": "#F24A72",
    "video_title": "#99AAB5",
    "video_id": "#FFFFFF",
    "start": "#00897B",
    "end": "#303F9F"
}

title = {
    "title" : "In Space with Markiplier\nAll routes",
    "font_file": "./Roboto/Roboto-Medium.ttf",
    "fontsize": 5,
    "alignment": "mm",
    "top": 80,
    "left": 80
}

video_title = {
    "font_file": "./Roboto/Roboto-Light.ttf",
    "fontsize": 1,
    "alignment": "mm"
}

video_id = {
    "font_file": "./Roboto/Roboto-Regular.ttf",
    "fontsize": 1,
    "alignment": "mm"
}

def magnitude(vector):
    """ Returns the magnitude of the vector.  """
    return np.linalg.norm(vector)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / magnitude(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def unit_perpendicular(v):
    """Returns the unit perpendicular vector to the given vector in the same plane. """
    x = np.array([0.0,0.0,1.0]) 
    v += [0.0]
    return unit_vector(np.cross(x,v))[0:2]

def ccw(A,B,C):
    """ Check if three points A B C are in counter clockwise direction """
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    """Check if two lines AB and CD intersect """
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

"""
Get the data from a pre created file nodes.nvm
Data is supposed to be in format {
                                   node_name: {
                                      details...,
                                      image: PILImageObject
                                      children: [node_name, ...] 
                                       },
                                   node_name: ... 
                                  }
Here my data is {
                  video_id: {
                      "id" : "..." ,
                      "title": "..." ,
                      "thumbnail": PILImageObject ,
                      "children": [video_id, ...]
                      }, ... }
"""
with open("nodes.nvm","rb") as f:
    data = pickle.load(f)

# Generate the network graph
G = nx.Graph()

# Add images in the graph, not actually added to the figure but added as data to G for later use
for node in data:
    G.add_node(node,image = data[node]["thumbnail"])

# Create a list of nodes and edges using data
nodes = list(data.keys())
edges = []
for node in data:
    for child in data[node]["children"]:
        edges+=[(node,child)]
        G.add_edge(node,child) 

# Finding Starts and Ends on the basis if the node has parents or childs
# No children = End
# No parents = Start
starts = []
ends = []
for node in nodes:
    isParent = isChild = False
    for parent, child in edges:
        if node == parent:
            isParent = True
        if node == child:
            isChild = True
    if isParent and not isChild:
        starts += [node]
    if isChild and not isParent:
        ends += [node]

# pos = {'j64oZLF443g': array([ 0.77145403, -0.57300887]), 'raIqPgW-quI': array([ 0.58869513, -0.62627967]), 'NK-u4ukbOFw': array([ 0.55949213, -0.39263847]), 'bYy_aAiTfYA': array([ 0.39484783, -0.4061523 ]), '7P9mpmsLSno': array([ 0.38824976, -0.21558735]), 'HcfjRwNr89Q': array([ 0.29763054, -0.11518634]), 'wKNzPHIk0EY': array([ 0.25556924, -0.00173218]), 'bjxEL2A9F4Q': array([0.27826083, 0.15238432]), 'Qbr2cyEgWS4': array([0.17245385, 0.09368269]), 'uoyvZ5mXiio': array([ 0.13649191, -0.02206534]), 'diaW8rX9CZg': array([0.04196613, 0.09007032]), 'YBDAQclN9jQ': array([-0.0762334 ,  0.00387218]), 'QgImksN6b3M': array([-0.17384706,  0.07568849]), '3De90tdLJmk': array([-0.27054469,  0.1983401 ]), 'NW-UYcqwDpU': array([-0.28859352,  0.35275463]), '3TboquFEMFA': array([-0.28928587,  0.50454579]), 'qeu7M8wIpyc': array([-0.21168085,  0.65768728]), '4zQJNqRjadQ': array([-0.41528548,  0.59454436]), 'fGewtUPe7TI': array([-0.5252656 ,  0.64424086]), 'PteUZUCJ7iY': array([-0.61456039,  0.56258646]), 'jGGT5FHDhFI': array([-0.14239939,  0.47054179]), 'iMz8rZD_9hg': array([-0.38860373,  0.22109669]), 'szDl_uKbOJg': array([-0.53962391,  0.28257017]), 'DsZDtqK4f1A': array([-0.70572164,  0.22786684]), '-Yn4Z-mPKMM': array([-0.81606642,  0.33509153]), 'MJM6QneZbVA': array([-0.85247263,  0.15054857]), '6cARNW6O4sY': array([-1.        ,  0.20932886]), 'g_ILOR7_mHw': array([-0.96122527,  0.03927965]), 'VpyFw2-Cec4': array([-0.57672844,  0.44764275]), 'cjGsjagqkk4': array([-0.49610577,  0.11736911]), 'aM-uQMUO6i4': array([-0.30388752, -0.03304188]), '17r42pf7kwY': array([-0.31538735,  0.0866643 ]), 'FuQt2BhgCQo': array([-0.28264221, -0.16667636]), '7iJoWgYwL7g': array([-0.41607504, -0.25465717]), 'eoOePnBbGWY': array([-0.31496724, -0.3211268 ]), 'wfdMicitgnA': array([-0.18444573, -0.04906642]), 'g72tvDIu_Ys': array([-0.03420424,  0.23712974]), 'uj8TRJDz98E': array([-0.15513156,  0.23293655]), '3cE9v0tdEGY': array([-0.16851026,  0.35239263]), 'CXUKjHzoMl4': array([ 0.00809349, -0.08262302]), 'Y0Ja_BrgGhA': array([ 0.31754867, -0.31360683]), 'uIMvjur42Vw': array([ 0.46990624, -0.31124585]), 'Dq1BrxAXQLg': array([ 0.4288049 , -0.52193415]), 'Tn1MkhNUaA4': array([ 0.28205289, -0.44886243]), 'ch07UcMzWkQ': array([ 0.2080399 , -0.26368293]), 'Rm7nK2x_FWQ': array([ 0.54933512, -0.51276321]), 'mGtFUm-sgh4': array([ 0.79431579, -0.40962204]), 'HHlphhgN1kU': array([ 0.71275559, -0.25188983]), 'ogrmyhb5gNI': array([ 0.62259631, -0.10849318]), 'SKODGzV20LU': array([ 0.47182291, -0.12898521]), '98bVPHiSY_k': array([0.58359109, 0.02657308]), 'uQO4CLQhKuY': array([ 0.3982033 , -0.03306016]), '5nZZAmvRIuU': array([0.43911858, 0.07995245]), '8Figj37SoPg': array([ 0.67890599, -0.37522468]), '5eG8rFt_JuY': array([ 0.66929308, -0.50816952])}

# Create pos charts (nodes with their coordinates on the axes) using library algorithms
# Replace with spring_layout, spectre_layout etc for other positions
pos = nx.kamada_kawai_layout(G)

def calc_distance(pos,G):
    """ Takes a pos chart and Graph object to return a DataFrame which contains distance between any two points in tabular form."""

    df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    for x in pos:
        for y in pos:
            df.loc[x,y] = np.linalg.norm(pos[x] - pos[y])
    return df


df = calc_distance(pos, G)

"""
This loop makes sure that two nodes are not near each other,
there is some gap between nodes, determined  by the variable: min_dist_bw_nodes
"""

# Loop variables and flag for breaking the loop
i = 1
j = 2
flag = 0 

while True:
    x = nodes[i-1]  # Node x
    y = nodes[j-1]  # Node y
    if flag == len(nodes)**2: # Means it will only stop when No changes in the DataFrame from the older one. That is if there are node^2 correct distances in the table.
        break
    
    if df[x][y] < min_dist_bw_two_nodes and x!=y:
        flag = 0            # Reset the flag because there has been change in the DataFrame
        y_x = pos[x] - pos[y]                                    # Vector from y to x
        difference = (min_dist_bw_two_nodes - df[x][y] + 0.001)  # Added a bit of difference for removing subtraction error
        
        l_x = l_y = 0         # l_x and l_y represent the distance ratio x and y will move apart from each other

        """
        Now having l_x , l_y = 1 , 1 would do the job but to take it one step ahead,
        we make it so that the two points move afar from each other in the ratio of space available behind them.
        The amount of space to consider in this calculation is done by the variable: node_clearance
        node_clearance should be greater than 1 for ideal cases.
    
        """
        
        for a in nodes:
            if df[x][a] < node_clearance*min_dist_bw_two_nodes and x!=a:  # If some node is near this node
                x_a = pos[a] - pos[x]
                if angle_between(x_a,y_x) < math.atan(node_clearance/0.5): # if that node is opposite to y and needs to be in consideration
                    l_x = min(df[x][a],l_x)
            
            if df[y][a] < node_clearance*min_dist_bw_two_nodes and y!=a:
                y_a = pos[a] - pos[y]
                if angle_between(y_a,y_x) < math.atan(node_clearance/0.5):
                    l_y = min(df[y][a],l_y)

       
        if not(l_y and l_x):    # If both nodes have clearance behind them to be moved
            l_x = l_y = 1

        pos[x] = pos[x] + unit_vector(y_x)*difference*(l_y/(l_x+l_y))
        pos[y] = pos[y] - unit_vector(y_x)*difference*(l_x/(l_x+l_y))
        df = calc_distance(pos,G)

    # Loop parameter increment
    else:
        flag +=1   
    if j == len(nodes):
        j=1
        if i == len(nodes):
            i=1
        else:
            i+=1 
    else:
        j+=1

# Create Figure on the Plot
fig, ax = plt.subplots(figsize = figure_size)


"""
normal coordinates = user defined whatever the data has the value of x and y, can be from like (-10 to 100), (-555 to -123) 
display coordinates = pixel coordinates on display, will be from (0 to width), (0 to height)
figure coordinates = from 0 to 1 in x and y 

"""

# Converts the data from normal coordinates to display coordinates
display_axis = ax.transData.transform

# Converst the data from display coordinates to figure coordinates
figure_coordinates = fig.transFigure.inverted().transform


# Disable the axes visibility
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xticks([])
plt.yticks([])

# Add Background Color
ax.set_facecolor(color_palette["background"])

# Remove space between axes and figure
plt.tight_layout(pad=0)


# Draw the start and end cards
length = min_dist_bw_two_nodes*card_length
width = min_dist_bw_two_nodes*card_width

for n in starts:
    x ,y =pos[n]
    x,y = x - width/2 , y - length/2
    rect = patches.Rectangle((x,y),width,length,fill = True,color = color_palette["start"])
    ax.add_patch(rect)

for n in ends:
    x ,y =pos[n]
    x,y = x - width/2 , y - length/2
    rect = patches.Rectangle((x,y),width,length,fill = True,color = color_palette["end"])
    ax.add_patch(rect)

    


# List recording the colors of the edges with index same as G.edges
edge_colors = [edge_color_list[0]] * len(edges)

# Recursive function that assigns colors the edges by checking if they intersect with each each other
def color_up(i):
    index_list = [a for a, x in enumerate(edge_colors) if x == edge_color_list[i]]

    if len(index_list) <= 1:
        return
    for index in index_list:
        for index2 in index_list:
            n,m = tuple(G.edges.keys())[index],tuple(G.edges.keys())[index2]
            A = tuple(pos[n[0]])
            B = tuple(pos[n[1]])
            C = tuple(pos[m[0]])
            D = tuple(pos[m[1]])
            temp = [A,B,C,D]
            if len(set(temp)) == len(temp)   and intersect(A,B,C,D) :
                edge_colors[index2] = edge_color_list[i+1]
                return color_up(i)
    return color_up(i+1)

color_up(0)

"""
The next loop makes the edges cool and curvy B) so that it doesnt come close to a node but curve around it.
It breaks a line into various lines and makes a curve out of it.
the variable used : min_edge_dist_from_node to know the distance the curve should be from the node.

"""
 
pseudo_edges = {}       # The list with the broken down lines 
for m,n in edges:       # Edge has two end points m and n
    ind =[{i,j} for i,j in list(G.edges.keys())].index({m,n}) # index of (m,n) in edge_colors
    x = [pos[m][0],pos[n][0]]     # list of x coordinates
    y = [pos[m][1],pos[n][1]]     # list of y coordinates
    x2 = x[:]
    y2 = y[:]
    
    """
    x and y will record the actual coordinates of the points on the curve
    x1 and y1 will record the coordinates of base of perpendiculars of points of curve on the line
    """

    for node in nodes:
        if node in (m,n):
            continue
        
        j = pos[m]              # j is coordinates of m
        k = pos[n]              # k is coordinates of n
        l = pos[node]           # l is coordinates of the node to be checked if it is close to the edge
        
        a = np.dot(unit_vector(l-j), unit_vector(k-j))  # cos angle  ljk
        b = np.dot(unit_vector(l-k), unit_vector(j-k))  # cos angle  lkj

        if a*b == 1:                                  # if l lies on line jk
            if magnitude(l-k) + magnitude(l-j) == magnitude(k-j):       # if l lies on line segment jk
                t = l + unit_perpendicular(k-j)*min_dist_bw_two_nodes*min_edge_dist_from_node       # t is the point on the curve
                t2 = l                                                                              # t2 is t projection on jk
                
                x += [t[0]]
                y += [t[1]]
                
                x2 += [t2[0]]
                y2 += [t2[1]]

        if a*b > 0:                                                      # if l lies on top or bottom oj jk
            d = magnitude(np.cross(k-j, l-j)/magnitude(k-j))             # perpendicular distance of l from jk
            if d < min_dist_bw_two_nodes*min_edge_dist_from_node :
               
                if np.dot(unit_perpendicular(k-j),l-k) > 0: 
                    t = l - unit_perpendicular(k-j)*(min_dist_bw_two_nodes*min_edge_dist_from_node)
                    t2 = l- unit_perpendicular(k-j)*d
                else: 
                    t = l + unit_perpendicular(k-j)*(min_dist_bw_two_nodes*min_edge_dist_from_node)
                    t2 = l + unit_perpendicular(k-j)*d
                
                x += [t[0]]
                y += [t[1]]
                
                x2 += [t2[0]]
                y2 += [t2[1]]

    x = [float(i) for i in x]     # numpy shit was giving warnings later on, converted to og
    y = [float(i) for i in y]

    """
    x3, y3 are the sorted x2, y2
    x4, y4 are the sorted x , y based on x3 and y3
    """
    x3 = x2[:]
    x3.sort()
    y3 = y2[:]
    y3.sort(reverse=(x[1]-x[0])*(y[1]-y[0]) < 0)

    x4 = []
    y4 = []
    for z, _x in enumerate(x3):
        x4.append(x[x2.index(_x)])
        y4.append(y[y2.index(y3[z])])
    
    x,y = np.array(x4),np.array(y4)
    

    # Cheat code. Just added some degree of color assuming it will intesect new lines once its curved. 
    # Cant be bothered to recheck the intesections with curve lines.
    edge_colors[ind] = edge_color_list[edge_color_list.index(edge_colors[ind]) + (len(x4) -2)//2 ]  

    """
    This part of the code just connects the points using magic and a little bit of algebra, neither of which i know, but the scipy library does.
    """

    is_sorted = lambda a: np.all(a[:-1] < a[1:]) or np.all(a[:-1] > a[1:])
    i, = np.where(y == pos[n][1]) 
    

    # Line mn 
    if i == 0:
        m, n = np.array([x[1],y[1]]),np.array([x[0],y[0]])
    else:
        m, n = np.array([x[-2],y[-2]]),np.array([x[-1],y[-1]])
    

    umn = unit_vector(n-m)
    umn_downscale =  umn*min_dist_bw_two_nodes*min_edge_dist_from_node
    n = n- umn_downscale

    x[i] = n[0]
    y[i] = n[1]

    k = len(y)
   

    # Curve line method 1 (spline)
    if is_sorted(x):
        xnew = np.linspace(x.min(), x.max(), 200) 
        spl = make_interp_spline(x, y, k=k-1)
        y_smooth = spl(xnew)
        ax.plot(xnew, y_smooth,color= edge_colors[ind])

    # Curve line method 2 (Cubic)
    elif k >2 :
        cubic_interploation_model = interp1d(y, x, kind = k-1)
        Y_=np.linspace(y.min(), y.max(), 1000)
        X_=cubic_interploation_model(Y_)
        ax.plot(X_, Y_,color= edge_colors[ind])
    
    # Curve **line** method 3
    else:
        ax.plot(x, y,color= edge_colors[ind])

    """
    Create arrows on the edges
    """
    ax.arrow(n[0],n[1],umn_downscale[0]/800,umn_downscale[1]/800,color = edge_colors[ind], head_width = min_dist_bw_two_nodes*arrow_distance/8 ,head_length = min_dist_bw_two_nodes*arrow_distance/4)



#? Add the respective image to each node

ax.get_xlim() # Dunno why but this makes the code work, maybe initialises the graph? god knows

length = node_length*min_dist_bw_two_nodes
width= node_width*min_dist_bw_two_nodes

for n in G.nodes:
    
    # Convert data coordinates to figure coordinates
    x,y = display_axis(pos[n])
    x,y = figure_coordinates((x,y))
    x,y = x - width/2, y - length/2
    a = plt.axes([x, y, width, length ]) # Plot mini axes on the nodes

    a.imshow(G.nodes[n]["image"]) # Put image on the axes

    # Distribute the title into lines
    s = data[n]["title"]
    s = s.split()
    def patcher(l, n):
        return "\n".join((" ".join(l[i:i+n]) for i in range(0, len(l), n)))
    s = patcher(s,3)
    
    # Hide the mini axes
    a.axis("off")

plt.savefig(image_name)

# plt.gcf().set_dpi(100) 
# plt.show(block=False)


"""
Post-Processing

Coz of the weird pixel measurments in matplotlib, using PIL to add text later.

"""

def display_coordinates(x):
    """ Returns pixel coordinates from data coordinates. """
    return fig.transFigure.transform(figure_coordinates(display_axis(x)))

# Gets canvas size
display_lim = fig.transFigure.transform((1,1))

# Converts a data length to pixels
def in_pixel(x):
    return display_coordinates((x,0))[0] - display_coordinates((0,0))[0] 

# Convert percentage in x direction to pixel
def per_x(x):
    return int(x*display_lim[0]/100)

# Convert percentage in y direction to pixel
def per_y(x):
    return int(x*display_lim[1]/100)

# Open Image
im = Image.open(image_name)
length = in_pixel(length)
width = in_pixel(width)
draw = ImageDraw.Draw(im)

# Initialise fonts
video_id_fontsize = per_y(video_id["fontsize"])
video_id_font = ImageFont.truetype(video_id["font_file"], video_id_fontsize)

video_title_fontsize = per_y(video_title["fontsize"])
video_title_font = ImageFont.truetype(video_title["font_file"], video_title_fontsize)


for m in G.nodes:
    # Add Video ID
    c = np.array([display_coordinates(pos[m])[0],display_lim[1] - display_coordinates(pos[m])[1]])
    c2 = c - np.array([0, length//2 + video_id_fontsize//2])
    draw.text(tuple(c2),m,color_palette["video_id"],font = video_id_font , anchor = video_id["alignment"])

    # Distribute the title into lines
    s = data[m]["title"]
    s = s.split()
    def patcher(l, n):
        return "\n".join((" ".join(l[i:i+n]) for i in range(0, len(l), n)))
    s = patcher(s,3)

    s = s.split("\n")

    # Add Video Title
    for ind,l in enumerate(s):
        c2 = c + np.array([0, length//2 + (2*ind+1)*video_title_fontsize//2])
        draw.text(tuple(c2),l,color_palette["video_title"],font = video_title_font , anchor = video_title["alignment"])

# Initialize title fonts
title_fontsize = per_y(title["fontsize"])
title_font = ImageFont.truetype(title["font_file"], title_fontsize)

# Add Title
c = np.array([per_x(title["left"]),per_y(title["top"])])
c = np.array([c[0],display_lim[1] - c[1]])
s = title["title"]
s = s.split("\n")

for ind,l in enumerate(s):
    c2 = c + np.array([0, (2*ind+1)*title_fontsize//2])
    draw.text(tuple(c2),l,color_palette["title"],font = title_font , anchor = title["alignment"])


# Final save
im.save(image_name)
