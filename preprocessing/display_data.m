function [ display_array ] = display_data( no_of_images, Images)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
if (no_of_images ~= size(Images,1))
    v = Images;   
    sel = randperm(size(v,1));
    sel = sel(1:no_of_images);
    X = v(sel,:);
else
    disp('here');
    X = Images;
end
example_width = 64;
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m n] = size(X);
example_height = (n / example_width);

% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :)));
        max_val = 1;
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m
		break; 
	end
end

% Display Image
figure;imshow(uint8(display_array));
% Do not show axis
axis image off

drawnow;


end

