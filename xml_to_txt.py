import os
import xml.etree.ElementTree as ET
import io
find_path = 'C:\pythonProject\study\other/'    #xml所在的文件
savepath='C:\pythonProject\study\other/'   #保存文件

class Voc_Yolo(object):
    def __init__(self, find_path):
        self.find_path = find_path
    def Make_txt(self, outfile):
        out = open(outfile,'w')
        print("创建成功：{}".format(outfile))
        return out
    def Work(self, count):
    #找到文件路径
        for root, dirs, files in os.walk(self.find_path):
        #找到文件目录中每一个xml文件
            for file in files:
            #记录处理过的文件
                count += 1
                #输入、输出文件定义
                input_file = find_path + file
                outfile = savepath+file[:-4]+'.txt'
                #新建txt文件，确保文件正常保存
                out = self.Make_txt(outfile)
                #分析xml树，取出w_image、h_image
                tree=ET.parse(input_file)
                root=tree.getroot()
                size=root.find('size')
                w_image=float(size.find('width').text)
                h_image=float(size.find('height').text)
                #继续提取有效信息来计算txt中的四个数据
                for obj in root.iter('object'):
                #将类型提取出来，不同目标类型不同，本文仅有一个类别->0
                    classname=obj.find('name').text
                    cls_id = classname
                    xmlbox=obj.find('bndbox')
                    x_min=float(xmlbox.find('xmin').text)
                    x_max=float(xmlbox.find('xmax').text)
                    y_min=float(xmlbox.find('ymin').text)
                    y_max=float(xmlbox.find('ymax').text)
                    #计算公式
                    x_center=((x_min+x_max)/2-1)/w_image
                    y_center=((y_min+y_max)/2-1)/h_image
                    w=(x_max-x_min)/w_image
                    h=(y_max-y_min)/h_image
                    #文件写入
                    out.write(str(cls_id)+" "+str(x_min)+" "+str(x_max)+" "+str(y_min)+" "+str(y_max)+'\n')
                out.close()
        return count
if __name__ == "__main__":
    data = Voc_Yolo(find_path)
    number = data.Work(0)
    print(number)