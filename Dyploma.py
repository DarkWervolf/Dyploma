import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join
import os
import plotly.graph_objects as go
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
import numpy.polynomial.polynomial as poly

import plotly.io as pio
pio.renderers.default = "browser"

def read_images_in_batches(path, batch_size=2000):
    # Get a list of files in the directory
    onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
    
    # Total number of files
    total_files = len(onlyfiles)
    
    # Iterate over the files in batches
    for start in range(0, total_files, batch_size):
        # Calculate the end of the batch
        end = min(start + batch_size, total_files)
        
        # Initialize the batch of images
        images = np.empty(end - start, dtype=object)
        
        # Load each image in the batch
        for i in range(start, end):
            # Read image in grayscale
            images[i - start] = cv2.imread(join(path, onlyfiles[i]), 0)
        
        # Yield the current batch of images
        yield images

def save_image(image, name, index, path):
    filename = name + str(index) + ".png"
    path = path + "/" + filename
    cv2.imwrite(path, image)

def detect_small_defects(image):
    image = cv2.equalizeHist(image)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(32, 32))
    image = clahe.apply(image)
    image = cv2.bilateralFilter(image, 9, 75, 75)

    image = cv2.GaussianBlur(image, (15, 15), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(image, 15, 150)

    # kernel = np.ones((17,17),np.uint8)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges

def plot_3d_surface_with_defects(data_matrix, defects_image, base_line, index, path):
    # Create a matrix that is initially set to constant_value everywhere
    surface_matrix = np.full_like(data_matrix, base_line, dtype=float)

    defects_image = defects_image[:defects_image.shape[0]-2,:]

    # Apply the defects mask: wherever defects_image is 1, we use data_matrix's values
    surface_matrix[defects_image == 255] = data_matrix[defects_image == 255]
    
    # Generate X and Y coordinates
    x = np.arange(data_matrix.shape[1])
    y = np.arange(data_matrix.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Create a 3D surface plot
    fig = go.Figure(data=[go.Surface(z=surface_matrix, x=X, y=Y)])
    
    # Update plot layout
    fig.update_layout(scene_aspectmode='manual',
                    scene_aspectratio=dict(x=data_matrix.shape[1]/100,
                                           y=data_matrix.shape[0]/100,
                                           z=1))
    
    filename = "3dplot_defects_" + str(index) + ".html"
    path = path + "/" + filename

    fig.write_html(path)

def get_ROI(image, relative):
    # if all the pixels are 0 (aka black) => abort
    thresh = 10
    im_bw = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    if cv2.countNonZero(im_bw) == 0:
        return 0, 0
  
    # The height and width of the image
    height, width = image.shape

    # Initialize an empty list to store the 3-pixel wide laser segments
    laser_segments = []
    centers = np.zeros((3, width), dtype=int)
    centers = []

    for x in range(width):
        # Find the rows in this column where the pixel is white
        white_pixels = np.where(image[:, x] > thresh)[0]

        if len(white_pixels) > 0:
            # Get the first and last white pixel in the column
            start, end = white_pixels[0], white_pixels[-1]

            # Calculate the central position
            center = start + (end - start) // 2
            # print(center, start, end)

            # Extract the 3-pixel wide segment
            segment_start = max(center - 1, 0)
            segment_end = min(center + 2, height)
            segment = image[segment_start:segment_end, x]

            centers.append(center)
            
            if len(segment) == 3:
                # Store the segment
                laser_segments.append(segment)
    
    if relative:
        centers = centers - np.median(centers)

    laser_segments = cv2.hconcat(laser_segments)
    if laser_segments.shape[1] != width:
        return 0, 0

    return laser_segments, centers

def interpolate_matrix(data_matrix):
    # Number of original rows and columns
    num_rows, num_cols = data_matrix.shape

    # New matrix with 3 times more rows
    # The last row will not have rows to interpolate if the original matrix has an odd number of rows
    expanded_matrix = np.zeros((3 * (num_rows - 1) + 1, num_cols))

    # Index for the new matrix
    new_index = 0

    # Loop through the original matrix rows
    for i in range(0, num_rows - 1):
        # Original row
        original_row = data_matrix[i]
        # Next row
        next_row = data_matrix[i + 1]

        # Place the original row in the new matrix
        expanded_matrix[new_index] = original_row
        new_index += 1

        # Calculate two new rows
        first_interpolated_row = (2 * original_row + next_row) / 3
        second_interpolated_row = (original_row + 2 * next_row) / 3

        # Place the interpolated rows in the new matrix
        expanded_matrix[new_index] = first_interpolated_row
        new_index += 1
        expanded_matrix[new_index] = second_interpolated_row
        new_index += 1

    # Ensure the last original row is also placed in the new matrix
    expanded_matrix[new_index] = data_matrix[-1]

    return expanded_matrix

def get_data(images, relative):
    a = []
    b = []
    line_width = -1
    for image in images:
        new_image, positions = get_ROI(image, relative)
        if hasattr(positions, "__len__"):
            a.append(positions)
            b.append(new_image)
            line_width = max(line_width, len(positions))

    if len(a) == 0:
        return None, None

    # data_matrix = cv2.vconcat(a)
    data_matrix = np.stack(a, axis=0)
    data_matrix = interpolate_matrix(data_matrix)
    stiched_image = cv2.vconcat(b)
    return data_matrix, stiched_image

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0o700)

def aggregate_matrix(data, block_size):
    """
    Aggregates a matrix by averaging over block_size x block_size blocks.
    This version adjusts the shape of the input data to be divisible by block_size.
    """
    # Reduce the size of the data array to be divisible by block_size
    new_height = data.shape[0] - data.shape[0] % block_size
    new_width = data.shape[1] - data.shape[1] % block_size
    data = data[:new_height, :new_width]
    
    # Calculate the new shape
    shape = (new_height // block_size, block_size,
             new_width // block_size, block_size)
    
    # Perform the aggregation
    return data.reshape(shape).mean(axis=(1, 3))

def save_3D_surface(data_matrix, index, path):
    # lower the resolution for visualisation
    # data_matrix = aggregate_matrix(data_matrix, 2)
    # Create x, y, and z coordinates for plotting
    x = np.arange(data_matrix.shape[1])  # Width of arrays
    y = np.arange(data_matrix.shape[0])  # Number of arrays
    z = data_matrix

    # Create a 3D surface plot
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(scene_aspectmode='manual',
                    scene_aspectratio=dict(x=data_matrix.shape[1]/100,
                                           y=data_matrix.shape[0]/100,
                                           z=1))
    
    filename = "3dplot_" + str(index) + ".html"
    path = path + "/" + filename

    fig.write_html(path)

def save_2D_surface(data_matrix, index, path):
    if data_matrix.dtype != np.uint8:
        data_matrix_color = data_matrix.astype(np.float32)  # Convert to float
        data_matrix_color = (data_matrix_color - data_matrix_color.min()) / (data_matrix_color.max() - data_matrix_color.min()) * 255  # Normalize to 0-255
        data_matrix_color = data_matrix_color.astype(np.uint8)  # Convert to unsigned 8-bit

    filename = "2dplot_" + str(index) + ".png"
    path_1 = path + "/" + filename
    data_matrix_color = cv2.applyColorMap(data_matrix_color, cv2.COLORMAP_PLASMA)
    cv2.imwrite(path_1, data_matrix_color)

def apply_dynamic_thresholds(image, thresholds, min_points=100):
    # Create an empty canvas for the zoned image
    zones = np.zeros_like(image)

    # Shades for zones, calculated based on the number of thresholds
    shades = np.linspace(50, 200, len(thresholds) + 1)

    # Assign initial zone with the first shade
    first_zone_mask = image < thresholds[0]
    zones[first_zone_mask] = shades[0]
    if np.count_nonzero(first_zone_mask) < min_points:
        zones[first_zone_mask] = 0  # Reset this zone if it has too few points

    # Iterate through thresholds to assign zones
    for idx, threshold in enumerate(thresholds):
        if idx < len(thresholds) - 1:
            # Create mask for the current threshold range
            mask = (image >= threshold) & (image < thresholds[idx + 1])
            zones[mask] = shades[idx + 1]
            if np.count_nonzero(mask) < min_points:
                zones[mask] = 0  # Reset this zone if it has too few points
        else:
            # Handle the last zone
            last_zone_mask = image >= threshold
            zones[last_zone_mask] = shades[-1]
            if np.count_nonzero(last_zone_mask) < min_points:
                zones[last_zone_mask] = 0  # Reset this zone if it has too few points

    return zones

def process_and_create_mask(data_matrix):
    blurred_matrix = cv2.GaussianBlur(data_matrix, (9, 9), 0)
    thresholds = np.linspace(39, 41, 30)
    zoned_image = apply_dynamic_thresholds(blurred_matrix, thresholds)
    return zoned_image

def adjust_values_to_match_overall_mean(data_matrix, zoned_image):
    overall_mean = np.median(data_matrix)
    unique_zones = np.unique(zoned_image)
    adjusted_matrix = np.copy(data_matrix)

    for zone in unique_zones:
        zone_indices = np.where(zoned_image == zone)
        zone_mean = np.median(data_matrix[zone_indices])
        adjustment = overall_mean - zone_mean
        adjusted_matrix[zone_indices] += adjustment

    return adjusted_matrix

#todo find threshold here
def adjust_data_matrix_to_zero_gradient(data_matrix, gradient_threshold=0):
    """
    Adjusts the data matrix to flatten the X and Y side projections if their gradients exceed a specified threshold.
    
    Args:
        data_matrix (np.array): The input data matrix representing Z values in a 3D depth map.
        gradient_threshold (float): The threshold for the gradient magnitude above which adjustments should be made.
        
    Returns:
        np.array: The adjusted data matrix, if necessary.
    """
    # Dimensions of the data_matrix
    n_rows, n_cols = data_matrix.shape

    # Process X-axis projections
    x_side_max = np.max(data_matrix, axis=0)
    x_side_min = np.min(data_matrix, axis=0)
    x_indices = np.arange(n_cols)
    x_max_coeffs = poly.polyfit(x_indices, x_side_max, 1)
    x_min_coeffs = poly.polyfit(x_indices, x_side_min, 1)

    print(abs(x_max_coeffs[1]), x_min_coeffs[1])

    # Check if adjustment is needed for X-axis
    if abs(x_max_coeffs[1]) > gradient_threshold or abs(x_min_coeffs[1]) > gradient_threshold:
        x_max_fit = poly.Polynomial(x_max_coeffs)
        x_min_fit = poly.Polynomial(x_min_coeffs)
        for i in range(n_cols):
            correction_slope = (x_max_fit(i) + x_min_fit(i)) / 2
            data_matrix[:, i] -= np.linspace(x_min_fit(i), x_max_fit(i), n_rows) - correction_slope

    # Process Y-axis projections
    y_side_max = np.max(data_matrix, axis=1)
    y_side_min = np.min(data_matrix, axis=1)
    y_indices = np.arange(n_rows)
    y_max_coeffs = poly.polyfit(y_indices, y_side_max, 1)
    y_min_coeffs = poly.polyfit(y_indices, y_side_min, 1)

    print(abs(y_max_coeffs[1]), y_min_coeffs[1])

    # Check if adjustment is needed for Y-axis
    if abs(y_max_coeffs[1]) > gradient_threshold or abs(y_min_coeffs[1]) > gradient_threshold:
        y_max_fit = poly.Polynomial(y_max_coeffs)
        y_min_fit = poly.Polynomial(y_min_coeffs)
        for i in range(n_rows):
            correction_slope = (y_max_fit(i) + y_min_fit(i)) / 2
            data_matrix[i, :] -= np.linspace(y_min_fit(i), y_max_fit(i), n_cols) - correction_slope

    return data_matrix

def save_side_projection(data_matrix, index, path):
    x_side_max = np.max(data_matrix, axis=0)
    x_side_min = np.min(data_matrix, axis=0)
    
    y_side_max = np.max(data_matrix, axis=1)
    y_side_min = np.min(data_matrix, axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(x_side_max)), y=x_side_max,
                             mode='lines', name='X-side Max'))
    fig.add_trace(go.Scatter(x=np.arange(len(x_side_min)), y=x_side_min,
                             mode='lines', name='X-side Min', fill='tonexty'))
    
    fig.add_trace(go.Scatter(x=np.arange(len(y_side_max)), y=y_side_max,
                             mode='lines', name='Y-side Max', xaxis='x2', yaxis='y2'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_side_min)), y=y_side_min,
                             mode='lines', name='Y-side Min', xaxis='x2', yaxis='y2', fill='tonexty'))

    fig.update_layout(
        xaxis=dict(domain=[0, 0.45]),
        yaxis=dict(title='Values'),
        xaxis2=dict(domain=[0.55, 1], anchor='y2'),
        yaxis2=dict(anchor='x2', title='Values'),
        title='2D Projections from X and Y sides (Max & Min)'
    )

    filename = "projections_" + str(index) + ".html"
    path = path + "/" + filename
    fig.write_html(path)

def calculate_stats(thickness_array):
    mean_diffs = [np.mean(diff) for diff in thickness_array]
    std_deviation = [np.std(diff) for diff in thickness_array]
    max_increases = [np.max(diff) for diff in thickness_array]
    min_decreases = [np.min(diff) for diff in thickness_array]
    

    return np.mean(mean_diffs), np.mean(std_deviation), np.mean(max_increases), np.mean(min_decreases)

def main():
    # defining constants
    thickness = 40 #in mm
    pix_to_mm_koef = 0.32 # 1pix == 0.32mm 

    # reading and stacking images
    # path = r'C:\Users\cosmo\Documents\Sync Docs\Dyploma\saveimage17\saveimage\1705954740'
    # path = r'C:\Users\cosmo\Documents\Sync Docs\Dyploma\saveimage14\saveimage\1706084016'
    # path = r'C:\Users\cosmo\Documents\Sync Docs\Dyploma\saveimage14\saveimage\1706090312'
    # path = r'C:\Users\cosmo\Documents\Sync Docs\Dyploma\saveimage12\saveimage\1706084037'
    # path = r'C:\Users\cosmo\Documents\Sync Docs\Dyploma\saveimage12\saveimage\1706090333'
    # path = r'C:\Users\cosmo\Documents\Sync Docs\Dyploma\saveimage19\saveimage\1706113129'
    
    # with marked defects
    path = r'C:\Users\cosmo\Documents\Sync Docs\Dyploma\saveimage18\saveimage\1706557455'

    
    thickness_difference_mm_rel_array = []
    thickness_difference_mm_zones_array = []
    thickness_difference_mm_rel_zones_array = []


    # testing dataset
    # path = r'C:\Users\cosmo\Documents\DyplomaImages\test_images'
    index = 0
    batch_size = 500
    for batch in read_images_in_batches(path, batch_size):
        data_matrix, stitched_image = get_data(batch, False)
        data_matrix_relative, stitched_image = get_data(batch, True)
        if data_matrix is None:
            print("Empty batch")
            continue

        folder_name = "DyplomaImages/out_1/"
        output_path = "C:/Users/cosmo/Documents/" + folder_name
        create_directory(output_path)

        save_image(stitched_image, "stitching_", index, output_path)

        folder_name = "DyplomaImages/out_1/real"
        output_path = "C:/Users/cosmo/Documents/" + folder_name
        create_directory(output_path)
        
        # defects = detect_small_defects(stitched_image)
        # save_image(defects, "defects_", index, output_path)

        overall_median = np.median(data_matrix)
        data_matrix = data_matrix - overall_median
        data_matrix_mm = data_matrix * pix_to_mm_koef + thickness

        data_matrix_relative = data_matrix_relative * pix_to_mm_koef + thickness

        # plot_3d_surface_with_defects(data_matrix_mm, defects, thickness, index, output_path)

        save_2D_surface(data_matrix_mm, index, output_path)
        save_3D_surface(data_matrix_mm, index, output_path)
        save_side_projection(data_matrix_mm, index, output_path)

        folder_name = "DyplomaImages/out_1/alternate/"
        output_path = "C:/Users/cosmo/Documents/" + folder_name
        create_directory(output_path)

        save_2D_surface(data_matrix_relative, index, output_path)
        save_3D_surface(data_matrix_relative, index, output_path)
        save_side_projection(data_matrix_relative, index, output_path)

        folder_name = "DyplomaImages/out_1/real_compensation/"
        output_path = "C:/Users/cosmo/Documents/" + folder_name
        create_directory(output_path)

        zones = process_and_create_mask(data_matrix_mm)
        data_matrix_fixed = adjust_values_to_match_overall_mean(data_matrix_mm, zones)

        save_2D_surface(data_matrix_fixed, index, output_path)
        save_3D_surface(data_matrix_fixed, index, output_path)
        save_side_projection(data_matrix_fixed, index, output_path)

        folder_name = "DyplomaImages/out_1/alternate_compensation/"
        output_path = "C:/Users/cosmo/Documents/" + folder_name
        create_directory(output_path)

        # data_matrix_fixed = adjust_data_matrix_to_zero_gradient(data_matrix_mm)
        

        zones_rel = process_and_create_mask(data_matrix_relative)
        data_matrix_fixed_rel = adjust_values_to_match_overall_mean(data_matrix_relative, zones_rel)

        save_2D_surface(data_matrix_fixed_rel, index, output_path)
        save_3D_surface(data_matrix_fixed_rel, index, output_path)
        save_side_projection(data_matrix_fixed_rel, index, output_path)

        
        thickness_difference_mm_rel = data_matrix_relative - data_matrix_mm
        thickness_difference_mm_rel_array.append(thickness_difference_mm_rel)


        thickness_difference_mm_zones = data_matrix_fixed - data_matrix_mm
        thickness_difference_mm_zones_array.append(thickness_difference_mm_zones)

        
        thickness_difference_mm_rel_zones = data_matrix_fixed_rel - data_matrix_mm
        thickness_difference_mm_rel_zones_array.append(thickness_difference_mm_rel_zones)


        index += 1
        print("Successfully processed %d batch" % index)
    
    mean_diff_rel, std_deviation_rel, max_increase_rel, max_decrease_rel = calculate_stats(thickness_difference_mm_rel_array)
    print("==============================")
    print("Relative VS Raw")
    print("mean difference = ", mean_diff_rel)
    print("std difference = ", std_deviation_rel)
    print("max increase = ", max_increase_rel)
    print("max decrease = ", max_decrease_rel)

    # Zones VS Raw
    mean_diff_zones, std_deviation_zones, max_increase_zones, max_decrease_zones = calculate_stats(thickness_difference_mm_zones_array)
    print("==============================")
    print("Zones VS Raw")
    print("mean difference = ", mean_diff_zones)
    print("std difference = ", std_deviation_zones)
    print("max increase = ", max_increase_zones)
    print("max decrease = ", max_decrease_zones)

    # Zones Rel VS Raw
    mean_diff_rel_zones, std_deviation_rel_zones, max_increase_rel_zones, max_decrease_rel_zones = calculate_stats(thickness_difference_mm_rel_zones_array)
    print("==============================")
    print("Zones Rel VS Raw")
    print("mean difference = ", mean_diff_rel_zones)
    print("std difference = ", std_deviation_rel_zones)
    print("max increase = ", max_increase_rel_zones)
    print("max decrease = ", max_decrease_rel_zones)

if __name__ == "__main__":
    main()

    print("End of program")
