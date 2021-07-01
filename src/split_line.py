from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage.util import invert
from PIL import Image as im
from heapq import *
import numpy as np
import matplotlib.pyplot as plt


def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1) 


#find the midway where we can make a threshold and extract the peaks regions
#divider parameter value is used to threshold the peak values from non peak values.
def find_peak_regions(hpp, divider=2):
    threshold = (np.max(hpp)-np.min(hpp))/divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks


#group the peaks into walking windows
def get_hpp_walking_regions(peaks_index):
    hpp_clusters = []
    cluster = []
    for index, value in enumerate(peaks_index):
        cluster.append(value)

        if index < len(peaks_index)-1 and peaks_index[index+1] - value > 1:
            hpp_clusters.append(cluster)
            cluster = []

        #get the last cluster
        if index == len(peaks_index)-1:
            hpp_clusters.append(cluster)
            cluster = []
            
    return hpp_clusters


#a star path planning algorithm 

def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                
    return []


def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    # global thresholding (much faster than local thresholding)
    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary*1
    return binary

def path_exists(window_image):
    #very basic check first then proceed to A* check
    if 0 in horizontal_projections(window_image):
        return True
    
    padded_window = np.zeros((window_image.shape[0],1))
    world_map = np.hstack((padded_window, np.hstack((window_image,padded_window)) ) )
    path = np.array(astar(world_map, (int(world_map.shape[0]/2), 0), (int(world_map.shape[0]/2), world_map.shape[1])))
    if len(path) > 0:
        return True
    
    return False

def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False
    
    for col in range(nmap.shape[1]):
        start = col
        end = col+20
        if end > nmap.shape[1]-1:
            end = nmap.shape[1]-1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
            road_blocks.append(col)

        if needtobreak == True:
            break
            
    return road_blocks

def group_the_road_blocks(road_blocks):
    #group the road blocks
    road_blocks_cluster_groups = []
    road_blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        road_blocks_cluster.append(value)
        if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

        if index == size-1 and len(road_blocks_cluster) > 0:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

    return road_blocks_cluster_groups

def extract_line_from_image(image, lower_line, upper_line):
    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.min(upper_line[:, 0])
    img_copy = np.copy(image)
    r, c = img_copy.shape
    return img_copy[lower_boundary:upper_boundary, :]    


def split_image(path_image):
  # img = rgb2gray(imread(path_image))
  # print(img)
  img = path_image
  # detects edges in image
  sobel_image = sobel(img)

  # Horizontal projection profile (HPP) is the array of the sum of rows of a 2D image. 
  # Where there are texts we see more peaks and the more white regions have a lower row sum. 
  # These peaks give us an idea of where the segmentation between two lines can be done.
  hpp = horizontal_projections(sobel_image)

  #mark potential region where image has to be split
  peaks = find_peak_regions(hpp)
  peaks_index = np.array(peaks)[:,0].astype(int)
  segmented_img = np.copy(img)
  r,c = segmented_img.shape
  for ri in range(r):
      if ri in peaks_index:
          segmented_img[ri, :] = 0

  #get all the regions where segmentation is to be made
  hpp_clusters = get_hpp_walking_regions(peaks_index)

  binary_image = get_binary(img)

  # Identify the regions where upper line text is connected to the lower line and make a cut in the middle
  for cluster_of_interest in hpp_clusters:
    if len(cluster_of_interest) > 10:
      nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
      road_blocks = get_road_block_regions(nmap)
      road_blocks_cluster_groups = group_the_road_blocks(road_blocks)
      #create the doorways
      for index, road_blocks in enumerate(road_blocks_cluster_groups):
          window_image = nmap[:, road_blocks[0]: road_blocks[1]+10]
          binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:][:, road_blocks[0]: road_blocks[1]+10][int(window_image.shape[0]/2),:] *= 0

  #now that everything is cleaner, its time to segment all the lines using the A* algorithm
  line_segments = []
  for i, cluster_of_interest in enumerate(hpp_clusters):
      nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
      path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
      offset_from_top = cluster_of_interest[0]
      if(len(path) != 0): 
        path[:,0] += offset_from_top
        line_segments.append(path)

  
  cluster_of_interest = hpp_clusters[1]
  offset_from_top = cluster_of_interest[0]
  nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
  path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))

  ## add an extra line to the line segments array which represents the last bottom row on the image
  last_bottom_row = np.flip(np.column_stack(((np.ones((img.shape[1],))*img.shape[0]), np.arange(img.shape[1]))).astype(int), axis=0)
  line_segments.append(last_bottom_row)

  line_images = []

  #save all images in "final" directory
  line_count = len(line_segments)
  # fig, ax = plt.subplots(figsize=(10,10), nrows=line_count-1)
  for line_index in range(line_count-1):
    line_image = extract_line_from_image(img, line_segments[line_index], line_segments[line_index+1])
    # print("line image",type(line_image))
    line_images.append(line_image)
    plt.imsave("../data/final/pic" + str(line_index) + ".png", line_image, cmap="gray")
  return line_images

# split_image("../data/pic.png")


