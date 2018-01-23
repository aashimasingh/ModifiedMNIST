function [ out ] = thresholdImage( input )
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
    input = uint8(input);
    out = zeros(size(input));
    histval = imhist(input);;
    sum = 0;
    count = 0;
    while (sum < 500)
        %disp('yhere');
        sum = sum + histval(256-count);
        count= count+1;
    end
    thres = 255-count+1;
    for r = 1:size(input,1)
        for c = 1:size(input,2)
            
            if (input(r,c) >= thres)
                %disp('here');
                out(r,c) = input(r,c);
            end
        end
    end
    
end

