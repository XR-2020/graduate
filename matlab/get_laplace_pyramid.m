function laplace_pyramid = get_laplace_pyramid(gaussian_pyramid, layer)
    w = (1/16) * [1, 4, 6, 4, 1];
    laplace_pyramid = cell(1, layer);
    laplace_pyramid{layer} = gaussian_pyramid{layer};
   
    for i=layer-1:-1:1
        resize_image = imresize(gaussian_pyramid{i+1}, 2, 'bilinear');
        first_filter = imfilter(resize_image, w, 'replicate');
        second_filter = imfilter(first_filter, w', 'replicate');
        laplace_pyramid{i} = gaussian_pyramid{i} - second_filter;
    end
end