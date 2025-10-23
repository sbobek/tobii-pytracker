import os
from psychopy import visual, core, monitors, event


def prepare_monitor(config):
    monitor_config = config.get_monitor_config()
    monitor = monitors.Monitor(name=monitor_config["name"])
    monitor.setWidth(monitor_config["width"])
    monitor.setDistance(monitor_config["distance"])
    monitor.setSizePix(monitor_config["resolution"])
    monitor.saveMon()

    return monitor


def prepare_window(config, monitor):
    window = visual.Window(
        fullscr=True, screen=0, size=monitor.getSizePix(),
        winType='pyglet', allowGUI=True, allowStencil=False,
        monitor=monitor.name, color='black',
        colorSpace='rgb', blendMode='avg', useFBO=True, units='pix')

    return window


def prepare_buttons(config, window, dataset):
    classes = dataset.get_classes()
    button_config = config.get_button_config()
    monitor_config = config.get_monitor_config()
    resolution_x, resolution_y = monitor_config["resolution"]

    num_buttons = len(classes)

    min_x, max_x = 300 - resolution_x/2, resolution_x - 300  # 300 pixels margin from left and right side of the screen
    fixed_y = 100 - resolution_y/2                           # 100 pixels margin from bottom of the screen

    default_width, default_height = button_config["size"]
    margin = button_config["margin"]

    total_width = max_x - min_x

    space_needed = num_buttons * default_width + (num_buttons - 1) * margin

    if space_needed > total_width:
        scale = total_width / space_needed
        button_width = default_width * scale
        actual_margin = margin * scale
    else:
        button_width = default_width
        actual_margin = margin

    button_height = default_height

    total_button_area = num_buttons * button_width + (num_buttons - 1) * actual_margin
    start_x = -total_button_area / 2 + button_width / 2

    color = button_config["color"]
    text_color = button_config["text"]["color"]

    buttons = []
    for i, class_name in enumerate(classes):
        x_pos = start_x + i * (button_width + actual_margin)
        y_pos = fixed_y
        label = class_name.lower()
        text_size = fit_text_to_area(window, class_name.upper(), button_width, button_height, button_config["text"]["size"])

        button = create_button(
            window=window,
            width=button_width,
            height=button_height,
            position=(x_pos, y_pos),
            text=class_name.upper(),
            label=label,
            fill_color=color,
            text_size=text_size,
            text_color=text_color
        )

        buttons.append(button)

    exit_button_text = "EXIT"
    exit_button_text_size = fit_text_to_area(window, exit_button_text, 100, 100, button_config["text"]["size"])
    exit_button_position = (resolution_x/2 - 100, resolution_y/2 - 100)  # 100 pixel margin from upper right screen corner
    exit_button = create_button(
        window=window,
        width=100,
        height=100,
        position=exit_button_position,
        text=exit_button_text,
        label="functional_quit",
        fill_color="red",
        text_size=exit_button_text_size,
        text_color="white"
    )

    buttons.append(exit_button)

    return buttons


def draw_window(config, window, data, is_text, buttons, focus_time, output_folder, border_width=5):
    button_config = config.get_button_config()
    area_x, area_y = config.get_area_of_interest_size()

    data_area = visual.Rect(
        win=window,
        width=area_x+border_width,
        height=area_y+border_width,
        pos=(0,0),
        fillColor='white',
        lineColor='white'
    )

    text = visual.TextStim(
        win=window,
        text="FOCUS HERE",
        pos=(0,35),
        height=25,
        color=button_config["text"]["color"],
        bold=False,
        antialias=True
    )

    x_text = visual.TextStim(
        win=window,
        text="X",
        pos=(0,0),
        height=button_config["text"]["size"],
        color="red",
        bold=True,
        antialias=True
    )

    data_area.draw()
    text.draw()
    x_text.draw()

    window.flip()
    core.wait(focus_time)

    for rect, text, _ in buttons:
        rect.draw()
        text.draw()

    if is_text:
        text_size = fit_text_to_area(window, data, area_x, area_y, 35)
        text = visual.TextStim(
            win=window,
            text=data,
            pos=(0,0),
            height=text_size,
            color="black",
            bold=False,
            antialias=True
        )

        data_area.draw()
        text.draw()
    else:
        image = visual.ImageStim(
            win=window,
            image=data,
            size=(area_x, area_y)
        )

        data_area.draw()
        image.draw()
    
    output_screenshot_path = save_screenshot(config, window, data, is_text, output_folder)

    window.flip()

    return output_screenshot_path


def create_button(window, width, height, position, text, label, fill_color, text_size, text_color, line_color="white", line_width=3, bold=True):
    rect = visual.Rect(
        win=window,
        width=width,
        height=height,
        pos=position,
        fillColor=fill_color,
        lineColor=line_color,
        lineWidth=line_width
    )

    text = visual.TextStim(
        win=window,
        text=text,
        pos=position,
        height=text_size,
        color=text_color,
        bold=bold
    )

    return (rect, text, label)


def fit_text_to_area(window, text, max_width, max_height, initial_size, min_size=10):
    size = initial_size

    while size >= min_size:
        stim = visual.TextStim(
            window,
            text=text,
            height=size,
            wrapWidth=None,
            units='pix'
        )

        bbox = stim.boundingBox

        if bbox[0] <= max_width * 0.9 and bbox[1] <= max_height * 0.9:
            return size

        size -= 1

    return min_size


def save_screenshot(config, window, data, is_text, output_folder):
    screenshot = window._getFrame(buffer='back')

    monitor_config = config.get_monitor_config()
    resolution_x, resolution_y = monitor_config["resolution"]
    area_x, area_y = config.get_area_of_interest_size()

    if is_text:
        screenshot_path = os.path.join(output_folder, data[:20]) + '.png'
    else:
        screenshot_path = os.path.join(output_folder, os.path.basename(data))

    crop_box = (
        resolution_x/2 - area_x/2,
        resolution_y/2 - area_y/2,
        resolution_x/2 + area_x/2,
        resolution_y/2 + area_y/2,
    )

    screenshot.crop(crop_box).save(screenshot_path, 'PNG')

    return screenshot_path



def show_instructions(window, text_lines, key='space'):
    """
    Display centered multi-line instruction text on a black background,
    and wait for the participant to press the specified key (default: SPACE).
    Automatically adapts wrap width and text size to the current window size.
    """
    # Get window size (width, height)
    win_width, win_height = window.size

    # Calculate text size and wrapping dynamically
    # About 1/25th of screen height looks readable
    text_height = win_height / 25
    wrap_width = win_width * 0.8  # leave margins on sides

    instruction_text = "\n\n".join(text_lines)

    message = visual.TextStim(
        win=window,
        text=instruction_text,
        color='white',
        height=text_height,
        wrapWidth=wrap_width,
        alignText='center',
        pos=(0, 0)
    )

    # Draw text and flip
    window.flip()  # clear screen
    message.draw()
    window.flip()

    # Wait for SPACE (or chosen key)
    event.waitKeys(keyList=[key])
    core.wait(0.2)