import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tobii_research as tr
import pandas as pd
import sys

# Global variables to store gaze data
gaze_data = {'left_gaze_point_on_display_area': (0.0, 0.0), 'right_gaze_point_on_display_area': (0.0, 0.0)}
trace = []

if len(sys.argv) > 1:
    savefile = sys.argv[1]
else:
    savefile = 'gaze.csv'

def gaze_data_callback(gaze_data_input):
    global gaze_data
    gaze_data = gaze_data_input
    trace.append(gaze_data_input)

def update_plot(frame, scatter):
    left_eye_mtplotlib_coord = (gaze_data['left_gaze_point_on_display_area'][0], 1-gaze_data['left_gaze_point_on_display_area'][1])
    right_eye_mtplotlib_coord = (gaze_data['right_gaze_point_on_display_area'][0], 1-gaze_data['right_gaze_point_on_display_area'][1])
    scatter.set_offsets([left_eye_mtplotlib_coord,right_eye_mtplotlib_coord])
    return scatter,
    
 
## Not tested, but shows how configureation gan be changed   
def configure_sampling_rate(eye_tracker, desired_rate):
    if 'GetEyeTrackerConfiguration' in eye_tracker.__dict__ and 'SetEyeTrackerConfiguration' in eye_tracker.__dict__:
        current_config = eye_tracker.GetEyeTrackerConfiguration()
        print("Current sampling rate: {} Hz".format(current_config['streaming_rate']))

        if current_config['streaming_rate'] != desired_rate:
            new_config = current_config.copy()
            new_config['streaming_rate'] = desired_rate
            eye_tracker.SetEyeTrackerConfiguration(new_config)
            print("Sampling rate set to {} Hz".format(desired_rate))
        else:
            print("Sampling rate is already set to {} Hz".format(desired_rate))
    else:
        print("The eye tracker does not support configuration settings.")

def main():
    eye_tracker = tr.find_all_eyetrackers()[0]

    if eye_tracker:
        print("Eye Tracker found: {}".format(eye_tracker.device_name))

        # Set up the matplotlib plot
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        scatter = ax.scatter([], [], label='Gaze Point', color='red')
        ax.legend()

        # Subscribe to gaze data
        gaze_subscription = eye_tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

        # Set up the animation
        ani = animation.FuncAnimation(fig, update_plot, fargs=(scatter,), blit=True)
        plt.show()

        try:
            input("Press Enter to exit...\n")
        finally:
            eye_tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
            pd.DataFrame(trace).to_csv(savefile, index=False)

    else:
        print("No Eye Tracker found.")

if __name__ == "__main__":
    main()
