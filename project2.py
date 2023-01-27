import os
import cv2

# đọc folder gốc
def initFolder():
   
    print("Start init folder")
    # init folder test- train
    train_fold_path ='image/train'
    test_fold_path =  'image/test'
    if not os.path.exists(train_fold_path):
        os.mkdir(train_fold_path)
    if not os.path.exists(test_fold_path):
        os.mkdir(test_fold_path)
        
    # init obj folder
    #     train
    #         obj1
    #         obj2 
    # 101
    for i in range (1,71):
        train_path = os.path.join('image/train',"obj"+str(i))
        test_path = os.path.join('image/test',"obj"+str(i))
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists( test_path):
            os.mkdir(test_path)
    print("End init folder")

#train : test = 8:2

def get_class_name(file_name):
    pos = file_name.find('_')
    res = file_name[:pos]
    return res

# phân chia vào folder train/test tỉ lệ 80/20
def initFile():
    file_names = os.listdir('coil-100')
    print(len(file_names))
    count = 0
    print("Start init file")
    for file_name in file_names:
        #print(os.path.join('coil-100', file_name))
        img = cv2.imread(os.path.join('coil-100', file_name))
        class_name = get_class_name(file_name)
        if count < 58:
            cv2.imwrite(os.path.join('image/train', class_name, file_name), img)
            #print(os.path.join('coil-100/train', class_name, file_name))
            count += 1
        else:
            cv2.imwrite(os.path.join('image/test', class_name, file_name), img)
            count += 1
            if count == 72:
                count = 0
    print("End init file")
 

