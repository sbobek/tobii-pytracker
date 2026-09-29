from tobii_pytracker.analyze.data_loader import DataLoader
from tobii_pytracker.configs.custom_config import CustomConfig
from pathlib import Path
import pandas as pd
config = CustomConfig('../configs/config.yaml')
loader = DataLoader(config, root='../')
print('Processing all samples with bbox generation + gaze mapping')

def setup_context():
    raw_sets = loader.get_all_data(flatten=False)
    gaze_data = loader.get_all_data(flatten=True)
    if not raw_sets:
        raise ValueError('No raw experiment data found.')

    if gaze_data.empty:
        raise ValueError('No gaze samples found in any run.')

    raw_data = pd.concat(raw_sets.values(), keys=raw_sets.keys(), names=['set_name', 'row_index']).reset_index(
        level='set_name').reset_index(drop=True)
    raw_data['slide_index'] = raw_data.groupby('set_name').cumcount()

    raw_data['set_name'] = raw_data['set_name'].astype(str)
    gaze_data['set_name'] = gaze_data['set_name'].astype(str)
    raw_data['slide_index'] = pd.to_numeric(raw_data['slide_index'], errors='coerce').astype('Int64')
    gaze_data['slide_index'] = pd.to_numeric(gaze_data['slide_index'], errors='coerce').astype('Int64')

    sets_with_gaze = set(gaze_data['set_name'].dropna().unique())
    raw_data = raw_data[raw_data['set_name'].isin(sets_with_gaze)].copy()
    gaze_data = gaze_data[gaze_data['set_name'].isin(sets_with_gaze)].copy()
    if raw_data.empty:
        raise ValueError('No raw rows remain after filtering runs with gaze.')

    gaze_counts = gaze_data.groupby(['set_name', 'slide_index']).size().rename('gaze_count').reset_index()
    candidate_rows = raw_data.merge(gaze_counts, on=['set_name', 'slide_index'], how='inner')
    candidate_rows = candidate_rows[candidate_rows['gaze_count'] > 0]
    if candidate_rows.empty:
        raise ValueError('No slide with gaze points was found.')

    print(f'Found {len(candidate_rows)} samples with gaze data')

    output_dir = Path('./analysis_outputs/bbox_all_samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    return gaze_data, candidate_rows, output_dir

def set_up_background_input(selected_raw, selected_gaze, method_bboxes):
    method_raw = pd.DataFrame([selected_raw.to_dict()])
    method_raw['set_name'] = method_raw['set_name'].astype(str)
    method_raw['slide_index'] = pd.to_numeric(method_raw['slide_index'], errors='coerce').astype('Int64')
    method_raw.at[0, 'objects_bboxes'] = {'image_bboxes': method_bboxes}

    background_data = pd.concat([
        method_raw.assign(avg_gaze_x=selected_gaze["avg_gaze_x"].iloc[j],
                          avg_gaze_y=selected_gaze["avg_gaze_y"].iloc[j])
        for j in range(len(selected_gaze))
    ]).reset_index(drop=True)
    return background_data

def set_up_bbox_method_context(dataset, input_image_path):
    generated_by_method = {
        'superpixel': dataset._detect_superpixels(str(input_image_path)),
        'grid': dataset._detect_grid(str(input_image_path)),
    }
    try:
        saliency_bboxes = dataset._detect_saliency(str(input_image_path))
        if saliency_bboxes:
            generated_by_method['saliency'] = saliency_bboxes
    except Exception as exc:
        print(f'  Saliency skipped: {exc}')

    for method, bboxes in generated_by_method.items():
        print(f'  {method}: {len(bboxes)} bboxes')
    return generated_by_method, method
