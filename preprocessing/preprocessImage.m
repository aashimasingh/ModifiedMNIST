function [ cleaned_data ] = preprocessImage( no_of_samples, data )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
    if (no_of_samples < size(data,1))
        sel = randperm(size(data,1));
        sel = sel(1:no_of_samples);
        X = data(sel,:);
    else
        X = data;
    end
    %impixelinfo;
    cleaned_data = zeros(size(X));
    image_width = 64;
    image_height = size(data,2)/image_width;
    %part 1
    for i = 1:size(X,1)
        sample = im2double(reshape(X(i,:),[image_height image_width]));

        sample_thresh = im2double(thresholdImage(sample));

        cleaned_data(i,:) = reshape(sample_thresh,[1 image_height*image_width]);
    end
   

    %part 2
    for i = 1:size(cleaned_data,1)
        sample = im2double(reshape(cleaned_data(i,:),[image_height image_width]));
        %

        %sample_b = imbinarize(sample);
        sample_bw = bwareaopen(sample, 20 ,8);

        cleaned_data(i,:) = reshape(sample_bw,[1 image_height*image_width]);
     end

end

