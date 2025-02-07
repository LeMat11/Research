% clear the workspace and variables
clc
clear all
close all

% reading the input images
imagea = imread('piv1.jpg');
imageb = imread('piv2.jpg');

% Convert to grayscale if they are RGB
if size(imagea,3)>1
    imagea = rgb2gray(imagea);
end
if size(imageb,3)>1
    imageb = rgb2gray(imageb);
end

% read the size of the actual image (same in this case)
[xmax,ymax] = size(imagea);

% window size (should be selected appropriately)
wsize = [40,40];
w_width = wsize(1);
w_height = wsize(2);

% define maximum displacement search
x_disp_max = w_width/2;
y_disp_max = w_height/2;

% define the grid for interrogation windows
xgrid = x_disp_max + w_width/2 + 1 : w_width/2 : xmax - (x_disp_max + w_width/2);
ygrid = y_disp_max + w_height/2 + 1 : w_height/2 : ymax - (y_disp_max + w_height/2);

% Range of 'search' windows in image 2
w_xcount = length(xgrid);
w_ycount = length(ygrid);

% pre-allocation
dpx = zeros(w_xcount,w_ycount);
dpy = zeros(w_xcount,w_ycount);

% Loop over each window
for i = 1:(w_xcount)
    for j = 1:(w_ycount)
        
        test_xmin = xgrid(i) - w_width/2;       % left x boundary of the interrogation window
        test_xmax = xgrid(i) + w_width/2;       % right x boundary
        test_ymin = ygrid(j) - w_height/2;      % bottom y boundary
        test_ymax = ygrid(j) + w_height/2;      % top y boundary
        
        % Extract interrogation windows
        test_ima = imagea(test_xmin:test_xmax, test_ymin:test_ymax);
        
        test_imb = imageb((test_xmin - x_disp_max):(test_xmax + x_disp_max), ...
                          (test_ymin - y_disp_max):(test_ymax + y_disp_max));
        
        % calculate the correlation
        correlation = normxcorr2(test_ima, test_imb);
        [xpeak, ypeak] = find(correlation == max(correlation(:)));
        
        % Re-scaling to original coordinates
        xpeak1 = test_xmin + xpeak - wsize(1)/2 - x_disp_max;
        ypeak1 = test_ymin + ypeak - wsize(2)/2 - y_disp_max;
        
        dpx(i,j) = xpeak1 - xgrid(i);
        dpy(i,j) = ypeak1 - ygrid(j);
        
        % -------- VISUALIZATION SECTION -----------

        % Show the interrogation window on imagea
        figure(1); clf;
        subplot(3,1,1)
        imshow(imagea, []);
        hold on;
        rectangle('Position', [test_ymin, test_xmin, w_height, w_width], 'EdgeColor', 'r', 'LineWidth', 1);
        title(sprintf('Interrogation Window on Image A (i=%d, j=%d)', i, j));
        hold off;

        % Show the search window on imageb
        subplot(3,1,2)
        imshow(imageb, []);
        hold on;
        % The search window is bigger: (test_xmin - x_disp_max):(test_xmax + x_disp_max) and similarly in y
        search_xmin = test_xmin - x_disp_max;
        search_ymin = test_ymin - y_disp_max;
        search_w = (w_width + 2*x_disp_max);
        search_h = (w_height + 2*y_disp_max);
        rectangle('Position', [search_ymin, search_xmin, search_h, search_w], 'EdgeColor', 'g', 'LineWidth', 1);
        title(sprintf('Search Window on Image B (i=%d, j=%d)', i, j));
        hold off;

        % Show the correlation result
        subplot(3,1,3)
        imshow(correlation, []);
        hold on;
        plot(ypeak, xpeak, 'ro');  % mark the peak
        title(sprintf('Correlation Result (i=%d, j=%d)', i, j));
        hold off;

        % To see the process step-by-step, you can pause or use drawnow
        %pause(0.1);  % pause for 0.1 seconds
        drawnow;

        % -----------------------------------------
        
    end
end
% display vector plot of the final displacement field
figure();
quiver(ygrid, xgrid, dpy', -dpx');
title('Displacement Field');

average_dpx = mean2(dpx);
average_dpy = mean2(dpy);
disp(['Average dpx: ', num2str(average_dpx)]);
disp(['Average dpy: ', num2str(average_dpy)]);