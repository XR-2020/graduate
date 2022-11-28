clear;
clc;
apple = double(imread('apple.jpeg'));
orange = double(imread('orange.jpeg'));
mask = double(zeros(512, 512, 3));
mask(:, 1:256, :) = 255;


layer = 6;
result_pyramid = cell(1, layer);

% 构造高斯金字塔
gaussian_apple = get_gaussian_pyramid(apple, layer);
gaussian_orange = get_gaussian_pyramid(orange, layer);
gaussian_mask = get_gaussian_pyramid(mask, layer);

% 构造拉普拉斯金字塔
laplace_apple = get_laplace_pyramid(gaussian_apple, layer);
laplace_orange = get_laplace_pyramid(gaussian_orange, layer);

% RES = A .* B + O .* (1 - B)
for i=1:layer
    result_pyramid{i} = laplace_apple{i} .* (gaussian_mask{i} / 255) + laplace_orange{i} .* (1 - (gaussian_mask{i} / 255));
end

% 重新构造图像 每次都是对上次的结果进行上采样，而不是最初的结果， 与前面从高斯构造拉普拉斯不同
w = (1/16) * [1, 4, 6, 4, 1];
for i=layer-1:-1:1
    resize_iamge = imresize(result_pyramid{i+1},2,'bilinear');
    first_filter = imfilter(resize_iamge, w, 'replicate');
    second_filter = imfilter(first_filter, w', 'replicate');
    result_pyramid{i} = result_pyramid{i} + second_filter;
end

% 显示图像
figure(1);
%{
for i=1:layer
    subplot(2,3,i);
    imshow(uint8(result_pyramid{i}));
end
%}
for i=1:1
    imshow(uint8(result_pyramid{i}));
end

