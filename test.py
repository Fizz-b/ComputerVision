 


import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

np.seterr(divide = 'ignore') 
def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def butterworthLP(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base

def remove_periodic_noise(img):
    # get the frequency domain
    fourier_transform = np.fft.fft2(img)
    fshift = np.fft.fftshift(fourier_transform)
    # show fourier transform
    fourier_noisy = 20*np.log(np.abs(fshift))

   # smoothen the vertical lines in the spatial domain =
   # remove the high frequency signals (i.e. horizontal lines) in the frequency domain 
  
    width,height = img.shape[:]
     # horizontal mask
     # horizontal mask
    fshift[width-5:width+6, 0:height-10] = 0
    fshift[width-5:width+6, height+11:] = 0
    
    # get the spatial domain back
    # inverse shift
    f_ishift = np.fft.ifftshift(fshift)
    # inverse furier
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back).astype(np.uint8)
   
    
    return img_back


# TODO : doc hieu xem ho lam gi denoisePeriodic
def denoisePeriodic(img, size_filter=2, diff_from_center_point=50,size_thresh=2):

    h,w = img.shape[:]
    img_float32 = np.fft.fft2(img)
    fshift = np.fft.fftshift(img_float32)

    #show the furier  image transform by log e of fft
    furier_tr = 20*np.log(np.abs(fshift))

    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(furier_tr, cmap = 'gray')
    # plt.show()

    #get center point value
    center_fur = furier_tr[int(h/2)][int(w/2)]

    #find pick freq point
    new_fur = np.copy(furier_tr)
    kernel = np.ones((2*size_filter+1,2*size_filter+1),np.float32)/((2*size_filter+1)*(2*size_filter+1)-1)
    kernel[size_filter][size_filter]=-1
    kernel = -kernel
    # print(kernel)
    dst = cv2.filter2D(new_fur,-1,kernel)

    # plt.subplot(121),plt.imshow(dst, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(dst, cmap = 'gray')
    # plt.subplot(122),plt.imshow(new_fur, cmap = 'gray')
    # plt.show()
    diff_from_center_point = center_fur*diff_from_center_point/356
    dst[0][:]=dst[1][:]=dst[:][0]=dst[:][1]=0

    dst[int(h/2)][int(w/2)]=0
    index = np.where(dst>diff_from_center_point)
    # print("index",index)

    # remove point isnot the pick one
    index_x = []
    index_y = []

    for i,item in enumerate(index[0]):
        
        value = furier_tr[index[0][i]][index[1][i]]
        # print("value ", value)
        matrix = np.copy(furier_tr[max(0,index[0][i]-size_filter):min(h,index[0][i]+size_filter+1),max(0,index[1][i]-size_filter):min(w,index[1][i]+size_filter+1)])
        # print("new maxtirx", matrix)
        matrix[size_filter][size_filter]=0
        
        max_value = np.amax(matrix)
        # print("mean", max_value)
        
        if (value-max_value<20):
            continue
        index_y.append(index[0][i])
        index_x.append(index[1][i])

        
    # print("to dau", index_x, index_y)
    # print("max freq", max_freq,center_fur)

    # set freq value of pick points to 1
    for i,item in enumerate(index_x):
        for j in range(size_thresh):
            for k in range(size_thresh):
                x = max(0,min(int(index_y[i]-int(size_thresh/2)+j),h-1))
                y = max(0,min(int(index_x[i]-int(size_thresh/2)+k),w-1))
                # print("toa do", x, y)
                furier_tr[x,y]=1
                fshift[x,y] = 1

    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(furier_tr, cmap = 'gray')

    # inverse to image
    # inverse shift
    f_ishift = np.fft.ifftshift(fshift)
    # inverse furier
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back).astype(np.uint8)
    # plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
    # plt.show()

    return img_back




# image 2 salt pepper
# image 4 low contrast
#A simple way to calculate contrast is by computing the standard deviation of the greyed image pixel intensities.
class Result:
    img = ""
    count = 0

    # The class "constructor" - It's actually an initializer
    def __init__(self, img, count):
        self.img = img
        self.count = count


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def count_obj(image):
    # convert gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape[:]
    thres = int(max(h,w)/160)
    
    # remove preiodic noise
    denoise = denoisePeriodic(gray, size_filter=int(thres/2), diff_from_center_point=50,size_thresh=thres)
   
    #denoise = remove_periodic_noise(gray)
    
 
    contrast = denoise.std()
    median = cv2.medianBlur(denoise,5)
    blur = cv2.GaussianBlur(median, (11,11), 0)
    
    # increase contrast
    if contrast< 40 :   
        equ = cv2.equalizeHist(blur)
    else:
     equ = blur
    # adaptive thresh hold
    thresh = cv2.adaptiveThreshold(equ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,199, 5)
 
    canny = cv2.Canny(thresh, 30, 100)
    dilated = cv2.dilate(canny, (1, 1), iterations=0)
    cnt, hierarchy = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    return cnt, rgb2

def show_images(images):
    rows = 1
    cols = 4
    for i in range(0, len(images), rows*cols):
        fig = plt.figure(figsize=(7, 8))
        for j in range(0, rows*cols):
            ax = fig.add_subplot(rows, cols, j+1)
            ax.set_title("Image "+str(j))
            plt.imshow(images[i+j])
    plt.show()


def show_result(images):
    rows = 1
    cols = 4
    for i in range(0, len(images), rows*cols):
        fig = plt.figure(figsize=(7, 8))
        for j in range(0, rows*cols):
            ax = fig.add_subplot(rows, cols, j+1)
            ax.set_title("Image  "+str(j)+":" +
                         str(images[i+j].count) + " objects")
            plt.imshow(images[i+j].img)
    plt.show()



def draw_contour(images):
    result = []
    for image in images:
        cnt, rgb2 = count_obj(image)
        print(len(cnt))
        res = Result(rgb2, len(cnt))
        result.append(res)
    return result


images = load_images_from_folder("test")

#show_images(images)

output = draw_contour(images)

show_result(output)
