function [ output ] = segmentUsingKmeansClass( image )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    [ row col] = size(image);
    output = zeros(3,28*28);
    %figure;imshow(image);
    transImage = transformDataForKmeans(image);
    [idx c] = kmeans(transImage,3);
    for i = 1:3
        group1 = transImage(idx==i,:);
        group = false(size(image));
        for s = 1:size(group1,1)
            group(group1(s,1),group1(s,2)) = 1;
        end
            %figure;imshow(group);
        filtImg = bwareafilt(group,1,'largest');
        filtImg = padarray(filtImg, [ 5 5 ]);
        bb = regionprops(filtImg,'BoundingBox');
        tlx = ceil(bb.BoundingBox(1));
        tly = ceil(bb.BoundingBox(2));
        xW = ceil(bb.BoundingBox(3));
        yW = ceil(bb.BoundingBox(4));
        temp = filtImg(tly:tly+yW,tlx:tlx+xW);
        temp = padarray(temp,[2 2]);
        temp = imresize(temp,[28 28]);
        output(i,:) = reshape(temp,[1 28*28]);
    end

end

