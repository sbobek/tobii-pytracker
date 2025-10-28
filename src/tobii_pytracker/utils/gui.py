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

    win_width, win_height = window.size
    num_buttons = len(classes)

    # Margins in pixels
    side_margin = 50          # margin from left/right edges
    bottom_margin = 50        # margin from bottom of screen
    button_height = button_config["size"][1]

    # Compute total width for buttons
    total_available_width = win_width - 2*side_margin
    button_width = min(button_config["size"][0], total_available_width / num_buttons)
    actual_margin = (total_available_width - button_width*num_buttons) / max(num_buttons-1,1)

    # Y position of buttons (near bottom)
    button_y = -win_height/2 + bottom_margin + button_height/2

    # X positions (span full width)
    start_x = -total_available_width/2 + button_width/2

    buttons = []
    text_color = button_config["text"]["color"]
    for i, class_name in enumerate(classes):
        x_pos = start_x + i * (button_width + actual_margin)
        label = class_name.lower()
        text_size = fit_text_to_area(window, class_name.upper(), button_width, button_height, button_config["text"]["size"])

        button = create_button(
            window=window,
            width=button_width,
            height=button_height,
            position=(x_pos, button_y),
            text=class_name.upper(),
            label=label,
            fill_color=button_config["color"],
            text_size=text_size,
            text_color=text_color
        )
        buttons.append(button)

    # Exit button (top-right)
    exit_button_size = 100
    exit_button_position = (win_width/2 - exit_button_size/2, win_height/2 - exit_button_size/2)
    exit_button_text_size = fit_text_to_area(window, "EXIT", exit_button_size, exit_button_size, button_config["text"]["size"])
    exit_button = create_button(
        window=window,
        width=exit_button_size,
        height=exit_button_size,
        position=exit_button_position,
        text="EXIT",
        label="functional_quit",
        fill_color="red",
        text_size=exit_button_text_size,
        text_color="white"
    )
    buttons.append(exit_button)

    return buttons



def draw_window(config, window, data, is_text, buttons, focus_time, output_folder, border_width=5):
    button_config = config.get_button_config()
    fixation_config = config.get_fixation_dot_config()
    area_x, area_y = config.get_area_of_interest_size()
    win_width, win_height = window.size

    # --- Stimulus area in center (black rectangle) ---
    data_area = visual.Rect(
        win=window,
        width=area_x + border_width,
        height=area_y + border_width,
        pos=(0, 0),
        fillColor='black',  # dark background
        lineColor='black'
    )
    data_area.draw()

    # --- White fixation dot in the center ---
    fixation_dot = visual.Circle(
        win=window,
        radius=fixation_config["size"],           # small, visible dot
        edges=32,
        fillColor=fixation_config["color"],
        lineColor=fixation_config["color"],
        pos=(0, 0)
    )
    fixation_dot.draw()

    window.flip()
    core.wait(focus_time)

    # --- Draw buttons at the bottom ---
    bottom_y = -win_height / 2 + 50  # 50 px from bottom
    num_buttons = len(buttons) - 1  # exclude exit button
    spacing = win_width / (num_buttons + 1)

    for idx, (rect, text, label) in enumerate(buttons[:-1]):  # skip exit button
        x_pos = -win_width / 2 + (idx + 1) * spacing
        rect.pos = (x_pos, bottom_y)
        text.pos = (x_pos, bottom_y)
        rect.draw()
        text.draw()

    # --- Exit button top-right ---
    exit_button = buttons[-1]
    rect, text, _ = exit_button
    rect.pos = (win_width / 2 - 60, win_height / 2 - 60)  # 60 px margin
    text.pos = (win_width / 2 - 60, win_height / 2 - 60)
    rect.draw()
    text.draw()

    # --- Draw stimulus again (image or text) ---
    if is_text:
        text_size = fit_text_to_area(window, data, area_x, area_y, 35)
        text_stim = visual.TextStim(
            win=window,
            text=data,
            pos=(0, 0),
            height=text_size,
            color="black",
            bold=False,
            antialias=True
        )
        data_area.draw()
        text_stim.draw()
    else:
        image_stim = visual.ImageStim(
            win=window,
            image=data,
            size=(area_x, area_y),
            pos=(0, 0)
        )
        data_area.draw()
        image_stim.draw()

    window.flip()
    output_screenshot_path = save_screenshot(config, window, data, is_text, output_folder)

    return output_screenshot_path


def save_screenshot(config, window, data, is_text, output_folder):
    screenshot = window._getFrame(buffer='front')

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
    #screenshot.save(screenshot_path, 'PNG')
    return screenshot_path


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