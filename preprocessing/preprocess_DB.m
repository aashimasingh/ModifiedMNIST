function [ final_output ] = preprocess_DB( trainx )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    [row col] = size(trainx);
    final_output = size([row*3 28*28]);
    cleared_image = preprocessImage(row,trainx);
    final_output = segmentData(cleared_image);
end

