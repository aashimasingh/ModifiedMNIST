function [ output , no_of_segments] = segmentImage( inputImage )
%UNTITLED16 Summary of this function goes here
%   Detailed explanation goes here
    %disp('there');
    [ r c] = size(inputImage);
    inputImage = logical(inputImage);
    segments = bwconncomp(inputImage, 8);
    output = zeros(3,28*28);
    no_of_segments = size(segments.PixelIdxList,2);
    if (no_of_segments == 3)
        for i = 1:no_of_segments
            letter = false(size(inputImage));
            letter(segments.PixelIdxList{i}) = true;
            imgnew = zeros(28);
            imgnew = alignImage(letter);
            size(imgnew);
            output(i,:) = reshape(imgnew,[1,(28*28)]);
        end
    else
        output = segmentUsingKmeansClass(inputImage);
    end
end

