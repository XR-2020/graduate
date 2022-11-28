% 抠图
%{
思想：
二值化前景图像，将标记为1的点的位置记录下来，相应的位置用前景替换背景
！注意示例所用图像为偏白色，所以要取反
%}
clear all;
clc;

% 读文件
front=imread('frontground.jpeg'); 
back=imread('background.jpeg');

% 将图片转化为double精度
front=im2double(front); % RGB类型是uint8转换成统一的double精度,方便计算

% 二值化前景图像
BW_front=im2bw(front,0.8); % 显示RGB二值化结果
BW_front=imcomplement(BW_front); % 二值化取反，因为背景是偏白色  ！注意

% 查找interest值
[a,b]=find(BW_front==1); % 找到前景图像索引值为1的位置 

%抠图：将二值化数值1标记像素块填补到背景中
for i=1:size(a,1)
    back(a(i),b(i),1)=front(a(i),b(i),1);%第一通道赋值
    back(a(i),b(i),2)=front(a(i),b(i),2);%第二通道赋值
    back(a(i),b(i),3)=front(a(i),b(i),3);%第二通道赋值
end

figure('NumberTitle', 'off', 'Name', '抠图结果');
imshow(back);title("抠图结果");

