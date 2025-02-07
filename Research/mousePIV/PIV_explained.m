% Clear the workspace, close all figures, and clear the Command Window
clc            % Clears the Command Window
clear all      % Removes all variables from the workspace
close all      % Closes all open figure windows

% Reading the input images
imagea = imread('piv1.jpg');  % Reads the first image from the file 'piv1.jpg' and stores it in 'imagea'
imageb = imread('piv2.jpg');  % Reads the second image from the file 'piv2.jpg' and stores it in 'imageb'

% Convert to grayscale if they are RGB
if size(imagea,3) > 1         % Checks if 'imagea' has more than one color channel (i.e., if it's RGB)
    imagea = rgb2gray(imagea);% Converts 'imagea' to grayscale if it is indeed RGB
end
if size(imageb,3) > 1         % Same check for 'imageb'
    imageb = rgb2gray(imageb);% Converts 'imageb' to grayscale if it is RGB
end

% Read the size of the image (assuming both are the same size)
[xmax,ymax] = size(imagea);    % 'xmax' = number of rows, 'ymax' = number of columns in 'imagea'

% Window size (user-defined). This is the size of the interrogation window.
wsize = [40,40];               % Interrogation window width = 40, height = 40
w_width = wsize(1);           % w_width = 40
w_height = wsize(2);          % w_height = 40

% Define maximum displacement search in each dimension
% This is how far the algorithm will look around the interrogation window in the second image.
x_disp_max = w_width/2;       % Maximum search in x-direction is half the window width
y_disp_max = w_height/2;      % Maximum search in y-direction is half the window height

% Define the grid for interrogation windows:
% We only consider window centers that do not exceed the image boundary.
xgrid = x_disp_max + w_width/2 + 1 : w_width/2 : xmax - (x_disp_max + w_width/2);
ygrid = y_disp_max + w_height/2 + 1 : w_height/2 : ymax - (y_disp_max + w_height/2);

% The number of interrogation windows in each direction
w_xcount = length(xgrid);     % Number of windows in x-direction
w_ycount = length(ygrid);     % Number of windows in y-direction

% Pre-allocate arrays for storing displacement results
dpx = zeros(w_xcount, w_ycount);  % Will hold displacements along the x-axis
dpy = zeros(w_xcount, w_ycount);  % Will hold displacements along the y-axis

% Loop over each interrogation window
for i = 1 : w_xcount
    for j = 1 : w_ycount
        
        % Calculate boundaries for the interrogation window in Image A
        test_xmin = xgrid(i) - w_width/2;   % Left x boundary of the window in Image A
        test_xmax = xgrid(i) + w_width/2;   % Right x boundary
        test_ymin = ygrid(j) - w_height/2;  % Bottom y boundary
        test_ymax = ygrid(j) + w_height/2;  % Top y boundary
        
        % Extract the interrogation window from Image A
        test_ima = imagea(test_xmin : test_xmax, test_ymin : test_ymax);
        
        % Define boundaries for the search region in Image B, 
        % which is the interrogation window plus the maximum displacement in each direction.
        test_imb = imageb( (test_xmin - x_disp_max) : (test_xmax + x_disp_max), ...
                           (test_ymin - y_disp_max) : (test_ymax + y_disp_max) );
        
        % Calculate the normalized cross-correlation between the two windows
        correlation = normxcorr2(test_ima, test_imb);
        
        % Find the coordinates of the peak in the correlation result
        [xpeak, ypeak] = find(correlation == max(correlation(:)));
        
        % Re-scale the peak coordinates back to the coordinate system of the original image
        xpeak1 = test_xmin + xpeak - wsize(1)/2 - x_disp_max;
        ypeak1 = test_ymin + ypeak - wsize(2)/2 - y_disp_max;
        
        % Compute the displacement by subtracting the interrogation window center (xgrid(i), ygrid(j))
        % from the peak coordinates (xpeak1, ypeak1).
        dpx(i,j) = xpeak1 - xgrid(i);
        dpy(i,j) = ypeak1 - ygrid(j);
        
        % -------- VISUALIZATION SECTION -----------
        
        % Show the interrogation window on Image A
        figure(1);          % Select figure 1 (create if it doesn't exist)
        clf;                % Clear the current figure
        subplot(3,1,1);     % Divide figure into a 3x1 grid, focus on the first cell
        imshow(imagea, []); % Display Image A
        hold on;
        % Draw a rectangle showing the interrogation window boundary
        rectangle('Position', [test_ymin, test_xmin, w_height, w_width], ...
                  'EdgeColor', 'r', 'LineWidth', 1);
        title(sprintf('Interrogation Window on Image A (i=%d, j=%d)', i, j));
        hold off;
        
        % Show the search window on Image B
        subplot(3,1,2);      % Move to the second cell in the 3x1 figure layout
        imshow(imageb, []);  % Display Image B
        hold on;
        % The search region is the larger area in Image B used for correlation
        search_xmin = test_xmin - x_disp_max;
        search_ymin = test_ymin - y_disp_max;
        search_w = (w_width + 2*x_disp_max);
        search_h = (w_height + 2*y_disp_max);
        rectangle('Position', [search_ymin, search_xmin, search_h, search_w], ...
                  'EdgeColor', 'g', 'LineWidth', 1);
        title(sprintf('Search Window on Image B (i=%d, j=%d)', i, j));
        hold off;
        
        % Show the correlation result
        subplot(3,1,3);      % Third cell in the 3x1 figure layout
        imshow(correlation, []);
        hold on;
        plot(ypeak, xpeak, 'ro');  % Mark the correlation peak with a red circle
        title(sprintf('Correlation Result (i=%d, j=%d)', i, j));
        hold off;
        
        % Force MATLAB to update the figures
        drawnow;
        
        % -----------------------------------------
        
    end
end

% Display the final displacement field as a vector plot
figure();                           % Create a new figure
quiver(ygrid, xgrid, dpy', -dpx');  % Quiver plot: note the transpose and sign to match coordinate orientation
title('Displacement Field');

% Calculate and display the average displacements
average_dpx = mean2(dpx);
average_dpy = mean2(dpy);
disp(['Average dpx: ', num2str(average_dpx)]);
disp(['Average dpy: ', num2str(average_dpy)]);