import os
import xml.etree.ElementTree as ET
import io
find_path = './Annotations/'    #xml所在的文件
savepath='./label/'   #保存文件
outfile = savepath +'one_hot_labels.txt'
out = open(outfile,'w')
out.write("filename"+"    "+"class"+'\n')
def class_to_num (classname):
    if classname == 'aeroplane':
        return 1
    elif classname == 'bicycle':
        return 2
    elif classname == 'bird':
        return 3
    elif classname == 'boat':
        return 4
    elif classname == 'bottle':
        return 5
    elif classname == 'bus':
        return 6
    elif classname == 'car':
        return 7
    elif classname == 'cat':
        return 8
    elif classname == 'chair':
        return 9
    elif classname == 'cow':
        return 10
    elif classname == 'diningtable':
        return 11
    elif classname == 'dog':
        return 12
    elif classname == 'horse':
        return 13
    elif classname == 'motorbike':
        return 14
    elif classname == 'person':
        return 15
    elif classname == 'pottedplant':
        return 16
    elif classname == 'sheep':
        return 17
    elif classname == 'sofa':
        return 18
    elif classname == 'train':
        return 19
    else:
        return 20


for root, dirs, files in os.walk(find_path):
    for file in files:
        one_hot=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        raw=file[:-4]+" "
        one_hot_raw=raw
        input_file = find_path + file
        tree = ET.parse(input_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            classname = obj.find('name').text
            num=class_to_num(classname)
            if one_hot[num-1] !=0:
                continue
            else:
                raw = raw + str(num) +" "
                one_hot[num-1] =1
        out.write(one_hot_raw+" "+str(one_hot).replace('[',"").replace("]","").replace(','," ")+'\n')



