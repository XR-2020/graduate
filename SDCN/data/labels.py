""""
person,bird,cat,cow,dog,horse,sheep,aeroplane,bicycle,boat,bus,car,motorbike,train,bottle,chair,dining table,potted plant,sofa,tvmonitor

"""

import os


#获取imgsets文件列表

dir_path=r'C:\task\dataset\PASCAL VOC\2007\VOCdevkit\VOC2007\ImageSets\Main'
img_sets=os.listdir(dir_path)
for file in img_sets:
    if '_train.txt' in file:
        train_file=open(os.path.join(dir_path,file))
        content=train_file.read()