function gaussian_pyramid = get_gaussian_pyramid(image, layer)
    w = (1/16) * [1, 4, 6, 4, 1];
    gaussian_pyramid = cell(1, layer);
    gaussian_pyramid{1} = image;
    for i=2:layer
        first_filter = imfilter(gaussian_pyramid{i-1}, w, 'replicate');
        second_filter = imfilter(first_filter, w', 'replicate');
        gaussian_pyramid{i} = second_filter(1:2:end, 1:2:end, :);
    end
end