import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import match_template

def main():
    # --- 1) READ INPUT IMAGES ---
    imagea = io.imread('piv1.jpg')  # or: cv2.imread('piv1.jpg', cv2.IMREAD_GRAYSCALE)
    imageb = io.imread('piv2.jpg')
    
    # Convert to grayscale if necessary
    if imagea.ndim == 3:
        imagea = color.rgb2gray(imagea)
    if imageb.ndim == 3:
        imageb = color.rgb2gray(imageb)

    # Convert to float32 (match_template often wants floating)
    imagea = imagea.astype(np.float32)
    imageb = imageb.astype(np.float32)

    # --- 2) DEFINE IMAGE & WINDOW PARAMETERS ---
    # image shape => (rows, cols) = (xmax, ymax)
    xmax, ymax = imagea.shape

    # Interrogation window size
    w_width  = 40  # number of rows in the interrogation window
    w_height = 40  # number of columns in the interrogation window

    # Max displacement
    x_disp_max = w_width  // 2
    y_disp_max = w_height // 2

    # --- 3) DEFINE THE GRID FOR INTERROGATION WINDOWS ---
    step_x = w_width // 2   # step in the row direction
    step_y = w_height // 2  # step in the column direction

    # For i in range-based indexing, we want to skip the edges by x_disp_max + w_width//2, etc.
    xgrid = np.arange(x_disp_max + w_width//2,  xmax - (x_disp_max + w_width//2),  step_x)
    ygrid = np.arange(y_disp_max + w_height//2, ymax - (y_disp_max + w_height//2), step_y)

    # Pre-allocate displacement arrays
    #   We'll store them as shape (#xgrid, #ygrid), matching your MATLAB approach.
    dpx = np.zeros((len(xgrid), len(ygrid)), dtype=np.float32)  # displacement in the row direction
    dpy = np.zeros((len(xgrid), len(ygrid)), dtype=np.float32)  # displacement in the column direction

    # --- 4) MAIN LOOP OVER INTERROGATION WINDOWS ---
    for i, xcenter in enumerate(xgrid):
        for j, ycenter in enumerate(ygrid):

            # Boundaries of the interrogation window in image A
            test_xmin = xcenter - w_width // 2
            test_xmax = xcenter + w_width // 2
            test_ymin = ycenter - w_height // 2
            test_ymax = ycenter + w_height // 2

            # Extract interrogation window from image A
            # Python slice ends are exclusive, so we do test_xmax+1
            templateA = imagea[test_xmin : test_xmax + 1, test_ymin : test_ymax + 1]

            # Determine the search window in image B
            search_xmin = test_xmin - x_disp_max
            search_xmax = test_xmax + x_disp_max
            search_ymin = test_ymin - y_disp_max
            search_ymax = test_ymax + y_disp_max

            # Clip boundaries to avoid going out of image range
            search_xmin = max(search_xmin, 0)
            search_xmax = min(search_xmax, xmax - 1)
            search_ymin = max(search_ymin, 0)
            search_ymax = min(search_ymax, ymax - 1)

            searchB = imageb[search_xmin : search_xmax + 1, search_ymin : search_ymax + 1]

            # --- 5) NORMALIZED CROSS-CORRELATION ---
            correlation = match_template(searchB, templateA, pad_input=False)
            
            # Get the location of the peak
            peak = np.unravel_index(np.argmax(correlation), correlation.shape)
            peak_x, peak_y = peak  # row, col in correlation array

            # --- 6) CONVERT PEAK BACK TO IMAGE COORDS ---
            # match_template returns the top-left corner of the best match
            # We want the center -> so we ADD half the template size
            xpeak1 = search_xmin + peak_x + (w_width  // 2)
            ypeak1 = search_ymin + peak_y + (w_height // 2)

            # Displacement relative to the center of the current interrogation window
            dpx[i, j] = xpeak1 - xcenter  # row displacement
            dpy[i, j] = ypeak1 - ycenter  # col displacement

            # OPTIONAL: If you want to visualize each iteration, uncomment:
            # ------------------------------------------------------------
            # fig, axes = plt.subplots(3, 1, figsize=(6, 8))
            #
            # # 1) Show the interrogation window on image A
            # axes[0].imshow(imagea, cmap='gray')
            # rectA = plt.Rectangle((test_ymin, test_xmin),
            #                       w_height, w_width,
            #                       edgecolor='r', fill=False, linewidth=1)
            # axes[0].add_patch(rectA)
            # axes[0].set_title(f'Interrogation Window A (i={i}, j={j})')
            #
            # # 2) Show the search window on image B
            # axes[1].imshow(imageb, cmap='gray')
            # rectB = plt.Rectangle((search_ymin, search_xmin),
            #                       (search_ymax - search_ymin + 1),
            #                       (search_xmax - search_xmin + 1),
            #                       edgecolor='g', fill=False, linewidth=1)
            # axes[1].add_patch(rectB)
            # axes[1].set_title(f'Search Window B (i={i}, j={j})')
            #
            # # 3) Show the correlation result
            # axes[2].imshow(correlation, cmap='gray')
            # axes[2].plot(peak_y, peak_x, 'ro')  # Mark the peak
            # axes[2].set_title('Correlation Peak')
            #
            # plt.tight_layout()
            # plt.show()
            # ------------------------------------------------------------
    
    # --- 7) DISPLAY THE FINAL DISPLACEMENT FIELD ---
    fig, ax = plt.subplots()

    # In MATLAB: quiver(ygrid, xgrid, dpy', -dpx')
    # We'll replicate that here by constructing:
    # X, Y from meshgrid(ygrid, xgrid), and use dpy.T, -dpx.T
    X, Y = np.meshgrid(ygrid, xgrid)  # X=cols, Y=rows
    U = dpy.T       # displacement in x-direction
    V = -dpx.T      # displacement in y-direction (with negative to match MATLAB orientation)

    ax.quiver(X, Y, U, V, color='red', width = 0.005)
    ax.set_title('Displacement Field')
    ax.invert_yaxis()  # so row=0 is at the top, like an image
    plt.xlim(20, 240)  # Set the range for the x-axis
    plt.ylim(20, 220)  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # --- 8) AVERAGE DISPLACEMENTS ---
    average_dpx = np.mean(dpx)
    average_dpy = np.mean(dpy)
    print(f'Average dpx: {average_dpx:.3f}')
    print(f'Average dpy: {average_dpy:.3f}')

if __name__ == "__main__":
    main()