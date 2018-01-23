function [ output, finder ] = segmentData( input )
%UNTITLED17 Summary of this function goes here
%   Detailed explanation goes here
    [ r , c ] = size(input);
    output = zeros(r*3,28*28);
    finder = zeros(1,100);
    for i = 1:r
        if (mod(i,500) == 0)
            i
        end
        sample = reshape(input(i,:),[64 64]);
        [ output_sing, segments] = segmentImage(sample);
        %size(output(i,:,:))
        %i
        %size(output_sing')
        finder(1,segments)= finder(segments)+1;
        comps = size(output_sing,1);
        for j = 1:comps
           output((i-1)*3+j,:) = output_sing(j,:);
        end
    end
end

