function [ output ] = transformDataForKmeans( image )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [r c] = size(image);
    output = zeros(size(1,2));
    size(output);
    count = 0;
    %disp('here');
    for i = 1:r
        for j = 1:c
            %disp('here');
            if (image(i,j)~=0)
                %disp('thh');
                count = count+1;
                output(count,1) = i;
                output(count,2) = j;
            end
        end
    end
end

