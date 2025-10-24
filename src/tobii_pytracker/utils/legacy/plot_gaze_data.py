import os
import csv
import ast
from PIL import Image
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt


"""
Main config object for the script
"""
CONFIG = None


"""
Reads YAML configuration file
Args:
    filename (str): Path to configuration file
Returns:
    dict: Parsed YAML configuration
Raises:
    FileNotFoundError: If the configuration file is not found
"""
def read_config(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Configuration file '{filename}' not found.")

    with open(filename, "r") as file:
        CONFIG = yaml.safe_load(file)

    return CONFIG


"""
Reads the CSV file and returns a list of rows
Args:
    input_csv (str): Path to the input CSV file
Returns:
    list: List of rows from the CSV file
"""
def read_csv(input_csv, delimiter=';'):
    rows = []
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        headers = next(reader)  # Get the header row
        for row in reader:
            rows.append(row)

    return headers, rows


"""
Parses gaze data from a string to a list of tuples
Args:
    gaze_data_str (str): Gaze data as a string
Returns:
    list: List of tuples representing gaze points
"""
def parse_gaze_data(gaze_data_str):
    gaze_data_str = ast.literal_eval(gaze_data_str)
    gaze_data = []

    for item in gaze_data_str:
        point = item[0]
        gaze_data.append(point)

    return gaze_data


"""
Plots gaze points on the image
Args:
    image (PIL.Image): The original image
    gaze_data (list): List of tuples representing gaze points
    output_path (str): Path to save the output image
"""
def plot_gaze_points(image, gaze_data, output_path, image_size, model_prediction):
    # image = image.resize((image_size, image_size), Image.LANCZOS)
    
    fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax.imshow(image, alpha=0.6)
    ax.axis('off')

    ax.set_ylim(ax.get_ylim()[::-1])

    x_min, y_min, x_max, y_max = model_prediction[2]
    rect_x = image_size/2 + x_min
    rect_y = image_size/2 + y_min
    rect_width = -x_min+x_max
    rect_height = -y_min+y_max

    filtered_gaze_data = [(x + image_size / 2, image_size / 2 - y) for x, y in gaze_data if x is not None and y is not None]

    inside = []
    outside = []

    for x, y in filtered_gaze_data:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            inside.append((x, y))
        else:
            outside.append((x, y))

    x_coords_inside = [point[0] for point in inside]
    y_coords_inside = [point[1] for point in inside]
    ax.scatter(x_coords_inside, y_coords_inside, c='green', s=10, alpha=0.6)

    x_coords_outside = [point[0] for point in outside]
    y_coords_outside = [point[1] for point in outside]
    ax.scatter(x_coords_outside, y_coords_outside, c='red', s=10, alpha=0.6)

    ax.text(10, 20, "Gaze History", color="white", fontsize=12,
        bbox=dict(facecolor="black", alpha=0.5))

    import matplotlib.patches as patches

    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

    fig.set_size_inches(image_size / 100, image_size / 100)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    saved_image = Image.open(output_path)
    flipped_image = saved_image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_image.save(output_path)

    print(f"Saved gaze points image to: {output_path}")


"""
Processes the CSV file to create images with gaze points for each row.

Args:
    input_csv (str): The path to the input CSV file.
    output_folder (str): The folder to save the output images.
"""
def process_csv(input_csv, output_folder, image_size):
    headers, rows = read_csv(input_csv)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for row in rows:
        image_path = row[0]
        gaze_data_str = row[4]
        model_prediction = ast.literal_eval(row[5])[0]
        gaze_data = parse_gaze_data(gaze_data_str)

        base_name = os.path.basename(row[0])

        gaze_data_output_path = os.path.join(output_folder, f"gaze_points_{base_name}")

        with Image.open(image_path) as img:
            plot_gaze_points(img, gaze_data, gaze_data_output_path, image_size, model_prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default="data.csv", help='Path to the input CSV file')
    parser.add_argument('--output_folder', type=str, default="output", help='Path to the output folder for resized images')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to YAML script config file')
    parser.add_argument('--image_size', type=int, default=750, help='Size to resize images and output')

    args = parser.parse_args()

    CONFIG = read_config(args.config_file)

    process_csv(args.input_csv, args.output_folder, args.image_size)
