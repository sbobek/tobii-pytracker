import csv
import ast
import matplotlib.pyplot as plt
from PIL import Image

image_size = 750
output_path = 'output.png'

headers = []
rows = []
with open('data2.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    headers = next(reader)  # Get the header row
    for element in reader:
        for row in element:
            rows.append(row)

gaze_data = []
gaze_data_str = ast.literal_eval(rows[4])
for item in gaze_data_str:
    gaze_data.append(item[0])

model_prediction = ast.literal_eval(rows[5])[0]
bounding_box = model_prediction[2]

with Image.open(rows[0]) as img:
    fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=130)
    ax.imshow(img, alpha=0.6)
    ax.axis('off')

    ax.set_ylim(ax.get_ylim()[::-1])

    filtered_gaze_data = [(x + image_size / 2, image_size / 2 - y) for x, y in gaze_data if x is not None and y is not None]

    inside = []
    outside = []

    x_min, y_min, x_max, y_max = bounding_box
    x_min = image_size/2 + x_min
    y_min = image_size/2 + y_min
    x_max = image_size/2 + x_max
    y_max = image_size/2 + y_max

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

    import matplotlib.patches as patches

    x_min, y_min, x_max, y_max = bounding_box
    rect_x = image_size/2 + x_min
    rect_y = image_size/2 + y_min
    rect_width = -x_min+x_max
    rect_height = -y_min+y_max

    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

    fig.set_size_inches(image_size / 100, image_size / 100)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    saved_image = Image.open(output_path)
    flipped_image = saved_image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_image.save(output_path)

    print(f"Saved gaze points image to: {output_path}")



from PIL import Image, ImageDraw, ImageFont

img = Image.open(output_path).convert("RGBA")

extra_height = 125
w, h = img.size

new_img = Image.new("RGBA", (w, h + extra_height), (255, 255, 255, 255))
new_img.paste(img, (0, 0))

draw = ImageDraw.Draw(new_img)
font = ImageFont.truetype("ARIAL.TTF", 28)

user_class = rows[3]
model_pred = model_prediction[0]
gaze_accuracy = len(inside) / (len(inside)+len(outside))
print(len(inside))
print(len(outside))

lines = [
    (f"user_classification = {user_class} | model_prediction = {model_pred}", "red"),
    (f"model_prediction_bbox = {bounding_box}", "blue"),
    (f"user_gaze_accuracy = {gaze_accuracy:.2f}", "green"),
]

y = h + 10
for line, color in lines:
    bbox = draw.textbbox((0, 0), line, font=font)
    text_w = bbox[2] - bbox[0]
    x = (w - text_w) // 2  # center align
    draw.text((x, y), line, font=font, fill=color)
    y += bbox[3] - bbox[1] + 10  # line spacing

new_img.save("updated_" + output_path)
