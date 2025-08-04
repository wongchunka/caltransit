import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.responses import FileResponse
import uvicorn
from scipy.signal import find_peaks
from fastapi.staticfiles import StaticFiles
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Configuration & Setup ---
COLUMN_MAPPING = {
    "video_filename": "Video Filename",
    "beats_per_minute": "Beats per Minute",
    "peak_to_peak_duration_ms": "Peak-to-Peak Duration (ms)",
    "amplitude_ratio_peak_baseline": "Amplitude Ratio (Peak/Baseline)",
    "time_to_peak_ms": "Time to Peak (ms)",
    "time_to_50_rise_ms": "Time to 50% Rise (ms)",
    "time_to_50_decay_ms": "Time to 50% Decay (ms)",
    "time_to_80_decay_ms": "Time to 80% Decay (ms)",
    "time_to_90_decay_ms": "Time to 90% Decay (ms)",
    "time_to_max_decay_rate_ms": "Time to Max Decay Rate (ms)",
    "max_decay_rate": "Max Decay Rate",
    "decay_fit_r_squared": "Decay Fit R-Squared",
    "velocity_magnitude_pixels_per_ms": "Velocity Magnitude (pixels/ms)",
    "velocity_x_pixels_per_ms": "Velocity X (pixels/ms)",
    "velocity_y_pixels_per_ms": "Velocity Y (pixels/ms)",
    "velocity_r_squared": "Velocity Fit R-Squared",
}

# Create directories if they don't exist
Path("uploads").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("static/frames").mkdir(exist_ok=True)


app = FastAPI()

# In-memory storage for batch sessions.
# In a production environment, you might replace this with Redis or another persistent store.
SESSIONS: Dict[str, Dict] = {}


app.mount("/static", StaticFiles(directory="static"), name="static")

# --- HTML Templates as Strings ---

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Calcium Transient Analysis</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
    </style>
</head>
<body>
    <h1>Calcium Transient Analysis</h1>
    <p>Upload one or more video files to begin analysis.</p>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="video_files" accept=".mov,.mp4,.avi" required multiple>
        <br><br>
        <input type="submit" value="Upload and Proceed to ROI Selection">
    </form>
</body>
</html>
"""

def calculate_velocity_from_activation_map(activation_map, mapping_mask, grid_h, grid_w, frame_interval):
    """
    Calculate overall velocity from activation map using linear regression.
    
    Parameters:
    - activation_map: 2D array with activation times (in frames)
    - mapping_mask: binary mask indicating valid regions
    - grid_h: height of each grid cell in pixels
    - grid_w: width of each grid cell in pixels 
    - frame_interval: time per frame in milliseconds
    
    Returns:
    - velocity_x: velocity in x direction (pixels/ms)
    - velocity_y: velocity in y direction (pixels/ms)
    - velocity_magnitude: overall velocity magnitude (pixels/ms)
    - r_squared: R-squared of the linear fit
    """
    
    # Get valid activation times and their spatial coordinates
    valid_points = []
    activation_times = []
    
    h, w = activation_map.shape
    
    for i in range(h):
        for j in range(w):
            if mapping_mask[i, j] == 1 and not np.isnan(activation_map[i, j]):
                # Convert grid indices to pixel coordinates (center of each grid)
                pixel_y = (i + 0.5) * grid_h  # Row index -> Y coordinate
                pixel_x = (j + 0.5) * grid_w  # Column index -> X coordinate
                activation_time_ms = activation_map[i, j] * frame_interval
                
                valid_points.append([pixel_x, pixel_y])
                activation_times.append(activation_time_ms)
    
    if len(valid_points) < 3:
        # Need at least 3 points for meaningful linear regression
        return 0.0, 0.0, 0.0, 0.0
    
    valid_points = np.array(valid_points)
    activation_times = np.array(activation_times)
    
    # Fit linear model: activation_time = a*x + b*y + c
    # This gives us the spatial gradient of activation time
    X = valid_points  # [x, y] coordinates
    y = activation_times  # activation times
    
    # Add intercept term
    X_with_intercept = np.column_stack([X, np.ones(len(X))])
    
    # Fit linear regression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_with_intercept, y)
    
    # Get coefficients: [a, b, c] where time = a*x + b*y + c
    coeffs = reg.coef_
    a, b = coeffs[0], coeffs[1]  # spatial gradients (dt/dx, dt/dy)
    
    # Calculate R-squared
    y_pred = reg.predict(X_with_intercept)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # The velocity magnitude is the inverse of the gradient of the activation time
    # and points in the same direction as the gradient (from early to late activation).
    # v = (1 / |grad(t)|^2) * grad(t), where speed is |v| = 1 / |grad(t)|.
    grad_t_squared = a**2 + b**2
    
    if grad_t_squared < 1e-12:  # Avoid division by zero for a flat plane.
        return 0.0, 0.0, 0.0, r_squared
        
    # Velocity components (distance/time) – signs corrected to align with propagation direction
    velocity_x = a / grad_t_squared   # pixels/ms
    velocity_y = b / grad_t_squared   # pixels/ms
    
    # Velocity magnitude (speed)
    velocity_magnitude = 1 / np.sqrt(grad_t_squared)
    
    return velocity_x, velocity_y, velocity_magnitude, r_squared

def generate_activation_map(video_qc, peaks, mean_peak_to_peak_duration, frame_interval, results_path, filename_only, f=50000):
    """
    Generate activation map from video data.
    
    Parameters:
    - video_qc: numpy array of shape (h_original, w_original, t) - video after quality control
    - peaks: array of peak time indices
    - mean_peak_to_peak_duration: mean duration between peaks in milliseconds
    - frame_interval: time per frame in milliseconds
    - results_path: path to save results
    - filename_only: filename for saving
    - f: number of grids (default 1000)
    """
    h_original, w_original, t = video_qc.shape
    
    # Calculate grid dimensions to approximate f total grids
    aspect_ratio = w_original / h_original
    h = int(np.sqrt(f / aspect_ratio))
    w = int(f / h)
    
    # Ensure at least 1x1 grid
    h = max(1, h)
    w = max(1, w)
    
    print(f"Creating {w}x{h} = {w*h} grids for activation map")
    
    # Calculate grid sizes
    grid_h = h_original // h
    grid_w = w_original // w
    
    # Initialize arrays
    signal_max = np.zeros((h, w))
    signal_min = np.full((h, w), np.inf)
    
    # Calculate min and max signals per grid
    for i in range(h):
        for j in range(w):
            # Define grid boundaries
            start_h = i * grid_h
            end_h = min((i + 1) * grid_h, h_original)
            start_w = j * grid_w
            end_w = min((j + 1) * grid_w, w_original)
            
            # Extract grid region across all time points
            grid_region = video_qc[start_h:end_h, start_w:end_w, :]
            
            # Calculate mean signal per time point for this grid
            mean_signals = np.mean(grid_region, axis=(0, 1))
            
            signal_max[i, j] = np.max(mean_signals)
            signal_min[i, j] = np.min(mean_signals)
    
    # Calculate signal range
    signal_range = signal_max - signal_min
    
    # Get median of signal range values
    signal_range_median = np.median(signal_range)
    
    # Create mapping mask
    mapping_mask = (signal_range >= signal_range_median).astype(int)
    
    # Save visualization matrices
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(signal_max, cmap='viridis')
    plt.title('Signal Max')
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    plt.imshow(signal_min, cmap='viridis')
    plt.title('Signal Min')
    plt.colorbar()
    
    plt.subplot(2, 3, 3)
    plt.imshow(signal_range, cmap='viridis')
    plt.title('Signal Range')
    plt.colorbar()
    
    plt.subplot(2, 3, 4)
    plt.imshow(mapping_mask, cmap='binary')
    plt.title('Mapping Mask')
    plt.colorbar()
    
    # Get peak times (need at least 3 peaks)
    if len(peaks) < 3:
        raise ValueError("Need at least 3 peaks for activation map generation")
    
    time_peak_1 = peaks[0]
    time_peak_2 = peaks[1] 
    time_peak_3 = peaks[2]
    
    # Subselect video from time_peak_1 to time_peak_2
    video_subset = video_qc[:, :, time_peak_1:time_peak_2]
    
    # Calculate global intensity values in filtered grids per time point
    global_intensities = []
    for t_idx in range(video_subset.shape[2]):
        frame = video_subset[:, :, t_idx]
        filtered_intensities = []
        
        for i in range(h):
            for j in range(w):
                if mapping_mask[i, j] == 1:
                    start_h = i * grid_h
                    end_h = min((i + 1) * grid_h, h_original)
                    start_w = j * grid_w
                    end_w = min((j + 1) * grid_w, w_original)
                    
                    grid_region = frame[start_h:end_h, start_w:end_w]
                    filtered_intensities.append(np.mean(grid_region))
        
        if filtered_intensities:
            global_intensities.append(np.median(filtered_intensities))
        else:
            global_intensities.append(0)
    
    # Find global minimum time point
    global_min_idx = np.argmin(global_intensities)
    time_start = time_peak_1 + global_min_idx
    
    # Convert mean_peak_to_peak_duration from ms to frames
    peak_to_peak_frames = int(mean_peak_to_peak_duration / frame_interval)
    
    # Identify activation time
    time_end = min(time_start + peak_to_peak_frames, t)
    activation_subset = video_qc[:, :, time_start:time_end]
    
    # Create activation map
    activation_map = np.full((h, w), np.nan)
    
    for i in range(h):
        for j in range(w):
            if mapping_mask[i, j] == 1:
                start_h = i * grid_h
                end_h = min((i + 1) * grid_h, h_original)
                start_w = j * grid_w
                end_w = min((j + 1) * grid_w, w_original)
                
                # Extract grid region across activation time window
                grid_region = activation_subset[start_h:end_h, start_w:end_w, :]
                
                # Calculate mean signal per time point for this grid
                mean_signals = np.mean(grid_region, axis=(0, 1))
                
                # Find time of peak signal
                peak_time_idx = np.argmax(mean_signals)
                activation_map[i, j] = time_start + peak_time_idx
    
    # Add activation map to the plot
    plt.subplot(2, 3, 5)
    # Calculate 10% and 90% percentiles for color scale, excluding NaN values
    valid_values = activation_map[~np.isnan(activation_map)]
    if len(valid_values) > 0:
        vmin = np.percentile(valid_values, 5)
        vmax = np.percentile(valid_values, 95)
    else:
        vmin, vmax = None, None
    im = plt.imshow(activation_map, cmap='jet', vmin=vmin, vmax=vmax)
    plt.title('Activation Map')
    plt.colorbar(im, label='Time (frames)')
    
    plt.tight_layout()
    
    # Save the combined visualization
    viz_path = results_path / f"{filename_only}_activation_analysis.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual matrices as separate images
    # Signal Max
    plt.figure(figsize=(8, 6))
    plt.imshow(signal_max, cmap='viridis')
    plt.title('Signal Max')
    plt.colorbar()
    plt.savefig(results_path / f"{filename_only}_signal_max.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Signal Min  
    plt.figure(figsize=(8, 6))
    plt.imshow(signal_min, cmap='viridis')
    plt.title('Signal Min')
    plt.colorbar()
    plt.savefig(results_path / f"{filename_only}_signal_min.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Signal Range
    plt.figure(figsize=(8, 6))
    plt.imshow(signal_range, cmap='viridis')
    plt.title('Signal Range')
    plt.colorbar()
    plt.savefig(results_path / f"{filename_only}_signal_range.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Mapping Mask
    plt.figure(figsize=(8, 6))
    plt.imshow(mapping_mask, cmap='binary')
    plt.title('Mapping Mask')
    plt.colorbar()
    plt.savefig(results_path / f"{filename_only}_mapping_mask.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate velocity from activation map
    velocity_x, velocity_y, velocity_magnitude, velocity_r_squared = calculate_velocity_from_activation_map(
        activation_map, mapping_mask, grid_h, grid_w, frame_interval
    )
    
    print(f"Calculated velocity: magnitude={velocity_magnitude:.4f} pixels/ms, "
          f"x={velocity_x:.4f}, y={velocity_y:.4f}, R²={velocity_r_squared:.4f}")
    
    # Activation Map with velocity arrows
    plt.figure(figsize=(10, 8))
    # Use the same percentile-based color scale for the individual activation map
    im = plt.imshow(activation_map, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title('Activation Map with Velocity Vector')
    plt.colorbar(im, label='Time (frames)')
    
    # Add velocity arrow if velocity is significant
    if velocity_magnitude > 0.001:  # Only show arrow if velocity is meaningful
        # Calculate arrow position (center of the map)
        center_y, center_x = activation_map.shape[0] / 2, activation_map.shape[1] / 2
        
        # Scale arrow length for visibility (normalize to map dimensions)
        arrow_scale = min(activation_map.shape) * 0.3  # 30% of the smaller dimension
        max_velocity_component = max(abs(velocity_x), abs(velocity_y))
        if max_velocity_component > 0:
            # Use the actual signed velocity components for arrow direction
            arrow_dx = (velocity_x / max_velocity_component) * arrow_scale
            arrow_dy = (velocity_y / max_velocity_component) * arrow_scale
            
            # Note: In image coordinates, y increases downward, so we need to flip dy
            plt.arrow(center_x, center_y, arrow_dx, arrow_dy, 
                     head_width=arrow_scale*0.1, head_length=arrow_scale*0.15, 
                     fc='red', ec='red', linewidth=2, alpha=0.8)
            
            # Add velocity text
            plt.text(0.02, 0.98, f'Velocity: {velocity_magnitude:.4f} pixels/ms\n'
                                f'Vx: {velocity_x:.4f}, Vy: {velocity_y:.4f}\n'
                                f'R²: {velocity_r_squared:.4f}',
                    transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(results_path / f"{filename_only}_activation_map.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save velocity data to a separate CSV file
    velocity_data = {
        'video_filename': filename_only,
        'velocity_magnitude_pixels_per_ms': velocity_magnitude,
        'velocity_x_pixels_per_ms': velocity_x,
        'velocity_y_pixels_per_ms': velocity_y,
        'velocity_r_squared': velocity_r_squared,
        'grid_height_pixels': grid_h,
        'grid_width_pixels': grid_w,
        'frame_interval_ms': frame_interval
    }
    
    velocity_df = pd.DataFrame([velocity_data])
    velocity_csv_path = results_path / f"{filename_only}_velocity_analysis.csv"
    velocity_df.to_csv(velocity_csv_path, index=False)
    
    print(f"Activation map analysis saved to {viz_path}")
    print(f"Individual maps saved as JPG files")
    print(f"Velocity analysis saved to {velocity_csv_path}")
    
    return activation_map, mapping_mask, signal_max, signal_min, signal_range, velocity_x, velocity_y, velocity_magnitude, velocity_r_squared

def get_batch_roi_html(
    batch_id: str,
    file_index: int,
    video_filename: str,
    frame_path: str,
    latest_plot_url: Optional[str],
    combined_table_html: str
):
    """Generates the HTML for the interactive batch processing page."""
    
    results_so_far_html = ""
    if latest_plot_url:
        # Also get the activation map URL for the previous video
        prev_video_name = latest_plot_url.split('/')[-1].replace('_transients_plot.png', '')
        latest_activation_url = f"/results/{batch_id}/{prev_video_name}_activation_map.jpg"
        
        results_so_far_html = f"""
        <h2>Latest Result</h2>
        <p>Plot for the previously analyzed video:</p>
        <img src="{latest_plot_url}" alt="Latest transient plot" style="max-width: 80%; height: auto; border: 1px solid #ccc; padding: 5px;">
        <p>Activation map for the previously analyzed video:</p>
        <img src="{latest_activation_url}" alt="Latest activation map" style="max-width: 80%; height: auto; border: 1px solid #ccc; padding: 5px;">
        <h2 style="margin-top: 2em;">Combined Results So Far</h2>
        <div class="table-container">{combined_table_html}</div>
        <hr style="margin: 2em 0;">
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Select ROI for Batch {batch_id}</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            #canvas-container {{
                position: relative;
                cursor: crosshair;
                width: fit-content;
                border: 1px solid #ccc;
            }}
            #frame-image, #draw-canvas {{
                position: absolute;
                top: 0;
                left: 0;
            }}
            #frame-image {{ z-index: 1; }}
            #draw-canvas {{ z-index: 2; }}
            .hidden {{ display: none; }}
            .results-table {{ border-collapse: collapse; margin: 25px 0; font-size: 0.9em; min-width: 400px; border-radius: 5px 5px 0 0; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); }}
            .results-table thead tr {{ background-color: #007bff; color: #ffffff; text-align: left; font-weight: bold; }}
            .results-table th, .results-table td {{ padding: 12px 15px; }}
            .results-table tbody tr {{ border-bottom: 1px solid #dddddd; }}
            .results-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
            .results-table tbody tr:last-of-type {{ border-bottom: 2px solid #007bff; }}
            .table-container {{ overflow-x: auto; }}
        </style>
    </head>
    <body>
        {results_so_far_html}

        <h1>Draw ROI for: <code>{video_filename}</code></h1>
        <p>Click and drag on the image to draw a freehand Region of Interest. You can only draw one ROI.</p>
        <p>This is file {file_index + 1} in the batch.</p>
        <div id="canvas-container">
            <img id="frame-image" src="{frame_path}" alt="First frame of video">
            <canvas id="draw-canvas"></canvas>
        </div>
        <br>
        <button id="analyze-btn">Analyze and Proceed</button>
        <button id="reset-btn">Reset ROI</button>

        <div id="loading" class="hidden">
            <h2>Processing...</h2>
            <p>Please wait. The analysis may take several minutes.</p>
        </div>

        <script>
            const frameImage = document.getElementById('frame-image');
            const canvas = document.getElementById('draw-canvas');
            const ctx = canvas.getContext('2d');
            const container = document.getElementById('canvas-container');
            let drawing = false;
            let points = [];

            frameImage.onload = () => {{
                canvas.width = frameImage.width;
                canvas.height = frameImage.height;
                container.style.width = frameImage.width + 'px';
                container.style.height = frameImage.height + 'px';
            }};

            canvas.addEventListener('mousedown', (e) => {{
                if (points.length > 0) return; // Prevent drawing multiple ROIs
                drawing = true;
                ctx.beginPath();
                ctx.moveTo(e.offsetX, e.offsetY);
                points.push({{x: e.offsetX, y: e.offsetY}});
            }});

            canvas.addEventListener('mousemove', (e) => {{
                if (!drawing) return;
                ctx.lineTo(e.offsetX, e.offsetY);
                ctx.strokeStyle = 'magenta';
                ctx.lineWidth = 2;
                ctx.stroke();
                points.push({{x: e.offsetX, y: e.offsetY}});
            }});

            canvas.addEventListener('mouseup', () => {{
                if (!drawing) return;
                drawing = false;
                ctx.closePath();
            }});

            document.getElementById('reset-btn').addEventListener('click', () => {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                points = [];
            }});

            document.getElementById('analyze-btn').addEventListener('click', async () => {{
                if (points.length < 3) {{
                    alert('Please draw a valid ROI before analyzing.');
                    return;
                }}
                document.getElementById('loading').classList.remove('hidden');

                const response = await fetch('/process', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        batch_id: "{batch_id}",
                        file_index: {file_index},
                        roi_points: points,
                    }})
                }});

                document.getElementById('loading').classList.add('hidden');
                if (response.ok) {{
                    const result = await response.json();
                    if (result.next_file_index !== null) {{
                        // Go to the next file in the batch
                        window.location.href = `/batch/${{result.batch_id}}/${{result.next_file_index}}`;
                    }} else {{
                        // Last file was processed, go to final results page
                        window.location.href = `/results/${{result.batch_id}}`;
                    }}
                }} else {{
                    const error = await response.json();
                    alert('An error occurred during analysis: ' + error.detail);
                }}
            }});
        </script>
    </body>
    </html>
    """


def get_results_html(folder_name: str, files: list, plot_files: list[str], activation_files: list[str], velocity_files: list[str], table_html: str):
    list_items = "".join(f'<li><a href="/results/{folder_name}/{f}" target="_blank">{f}</a></li>' for f in files)

    plots_html_content = ""
    if plot_files:
        plots_html_content += "<h2>Transient Plots</h2>"
        for plot_file in plot_files:
            plot_url = f"/results/{folder_name}/{plot_file}"
            plots_html_content += f'<div><h3 style="margin-top: 1.5em;">{plot_file}</h3><img src="{plot_url}" alt="Transient plot: {plot_file}" style="max-width: 100%; height: auto;"></div>'

    activation_html_content = ""
    if activation_files:
        activation_html_content += "<h2>Activation Map Analysis</h2>"
        for activation_file in activation_files:
            activation_url = f"/results/{folder_name}/{activation_file}"
            activation_html_content += f'<div><h3 style="margin-top: 1.5em;">{activation_file}</h3><img src="{activation_url}" alt="Activation analysis: {activation_file}" style="max-width: 100%; height: auto;"></div>'

    velocity_html_content = ""
    if velocity_files:
        velocity_html_content += "<h2>Velocity Analysis Files</h2>"
        velocity_html_content += "<p>Individual velocity analysis CSV files for each video:</p>"
        velocity_html_content += "<ul>"
        for velocity_file in velocity_files:
            velocity_url = f"/results/{folder_name}/{velocity_file}"
            velocity_html_content += f'<li><a href="{velocity_url}" target="_blank">{velocity_file}</a></li>'
        velocity_html_content += "</ul>"

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Results for {folder_name}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 2em; line-height: 1.6; background-color: #f8f9fa; color: #212529; }}
            .container {{ max-width: 960px; margin: auto; background: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #343a40; }}
            h1 {{ border-bottom: 2px solid #dee2e6; padding-bottom: 0.5em; }}
            code {{ background-color: #e9ecef; padding: .2em .4em; margin: 0; font-size: 85%; border-radius: 3px; }}
            img {{ border: 1px solid #dee2e6; border-radius: 4px; padding: 5px; max-width: 100%; height: auto; margin-top: 1em; }}
            .results-table {{ border-collapse: collapse; margin: 25px 0; font-size: 0.9em; min-width: 400px; border-radius: 5px 5px 0 0; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); }}
            .results-table thead tr {{ background-color: #007bff; color: #ffffff; text-align: left; font-weight: bold; }}
            .results-table th, .results-table td {{ padding: 12px 15px; }}
            .results-table tbody tr {{ border-bottom: 1px solid #dddddd; }}
            .results-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
            .results-table tbody tr:last-of-type {{ border-bottom: 2px solid #007bff; }}
            .table-container {{ overflow-x: auto; }}
            .downloads {{ margin-top: 2em; }}
            .downloads ul {{ list-style: none; padding-left: 0; }}
            .downloads li a {{ text-decoration: none; color: #007bff; }}
            .downloads li a:hover {{ text-decoration: underline; }}
            .home-link {{ display: inline-block; margin-top: 2em; padding: 10px 15px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
            .home-link:hover {{ background-color: #0056b3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Analysis Results for Batch <code>{folder_name}</code></h1>
            
            <h2 style="margin-top: 2em;">Combined Summary Data</h2>
            <div class="table-container">{table_html}</div>

            {plots_html_content}

            {activation_html_content}

            {velocity_html_content}
    
            <div class="downloads">
                <h2>Download Files</h2>
                <ul>{list_items}</ul>
            </div>

            <a href="/" class="home-link">Analyze more videos</a>
        </div>
    </body>
    </html>
    """


# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return INDEX_HTML

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(
    video_files: List[UploadFile] = File(...)
):
    batch_id = str(uuid.uuid4())
    upload_dir = Path("uploads") / batch_id
    upload_dir.mkdir(exist_ok=True)
    results_dir = Path("results") / batch_id
    results_dir.mkdir(exist_ok=True)

    filenames = []
    for video_file in video_files:
        video_filename = video_file.filename
        video_path = upload_dir / video_filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        filenames.append(video_filename)

    SESSIONS[batch_id] = {
        "files": filenames,
        "results_data": [],  # This will store a list of result dicts
    }

    # Redirect to the batch processing page for the first file
    return RedirectResponse(url=f"/batch/{batch_id}/0", status_code=303)

@app.get("/batch/{batch_id}/{file_index}", response_class=HTMLResponse)
async def batch_roi_page(batch_id: str, file_index: int):
    """Serves the page for selecting ROI for a file in a batch."""
    if batch_id not in SESSIONS:
        return HTMLResponse("Batch session not found.", status_code=404)

    session = SESSIONS[batch_id]
    files = session['files']

    if file_index >= len(files):
        # All files in this batch have been processed.
        return RedirectResponse(url=f"/results/{batch_id}", status_code=303)

    video_filename = files[file_index]
    video_path = Path("uploads") / batch_id / video_filename

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return HTMLResponse(f"Error: Could not open video file {video_filename}.", status_code=500)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return HTMLResponse(f"Error: Could not read first frame of {video_filename}.", status_code=500)

    # Save frame to a static dir for the browser to access
    static_batch_dir = Path("static/frames") / batch_id
    static_batch_dir.mkdir(exist_ok=True)
    frame_filename = f"{file_index}.png"
    frame_path_server = static_batch_dir / frame_filename
    cv2.imwrite(str(frame_path_server), frame)
    frame_path_client = f"/static/frames/{batch_id}/{frame_filename}"

    # Get results from previous step to display
    latest_plot_url = None
    combined_table_html = "<p>No results yet. This is the first file.</p>"
    if file_index > 0 and session['results_data']:
        prev_video_name = Path(files[file_index - 1]).stem
        latest_plot_url = f"/results/{batch_id}/{prev_video_name}_transients_plot.png"
        df = pd.DataFrame(session['results_data'])
        df_display = df.rename(columns=COLUMN_MAPPING)
        combined_table_html = df_display.to_html(index=False, classes='results-table')

    return HTMLResponse(get_batch_roi_html(
        batch_id=batch_id,
        file_index=file_index,
        video_filename=video_filename,
        frame_path=frame_path_client,
        latest_plot_url=latest_plot_url,
        combined_table_html=combined_table_html
    ))


@app.post("/process")
async def process_video(request: Request):
    """
    This is the main analysis logic.
    It receives the ROI for one video in a batch and processes it.
    """
    try:
        data = await request.json()
        batch_id = data['batch_id']
        file_index = data['file_index']
        roi_points = data['roi_points']

        if batch_id not in SESSIONS:
            raise ValueError("Invalid batch session.")

        session = SESSIONS[batch_id]
        video_filename = session['files'][file_index]
        
        video_path = Path("uploads") / batch_id / video_filename
        filename_only = video_path.stem
        
        # --- Create results directory ---
        results_path = Path("results") / batch_id
        results_path.mkdir(exist_ok=True)


        # --- Video Loading and Validation ---
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_rate < 20:
            raise ValueError(f"{video_filename} not analysed: frame rate is less than 20 FPS.")
        
        # Original script removes first 5 and last 5 frames
        if frame_count < 90: # original was 100, but 10 frames are removed
             raise ValueError(f"{video_filename} not analysed: less than 100 frames total.")

        frame_count_adjusted = frame_count - 10
        frame_interval = 1000 / frame_rate

        # --- Create Mask from ROI ---
        mask = np.zeros((video_height, video_width), dtype=np.uint8)
        roi_contour = np.array([[[p['x'], p['y']]] for p in roi_points], dtype=np.int32)
        cv2.fillPoly(mask, [roi_contour], 255)

        if np.sum(mask > 0) <= 100:
            raise ValueError("Not enough selection in ROI. Please draw a larger area.")

        # --- Signal Extraction ---
        raw_signal = np.zeros(frame_count_adjusted)
        
        # Create video_qc array to store the entire video for activation map analysis
        video_qc = np.zeros((video_height, video_width, frame_count_adjusted), dtype=np.uint8)
        
        # Skip first 5 frames
        for _ in range(5):
            cap.read()

        for t in range(frame_count_adjusted):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Store frame in video_qc array for activation map analysis
            video_qc[:, :, t] = frame_bw
            
            # Calculate mean intensity within the ROI
            mean_intensity = cv2.mean(frame_bw, mask=mask)[0]
            raw_signal[t] = mean_intensity
        
        cap.release()

        # --- Denoising ---
        # wdenoise in MATLAB with sym4, level 5, BlockJS is complex to replicate 1:1.
        # pywavelets denoise_wavelet is a good equivalent.
        signal = denoise_wavelet(raw_signal, wavelet='sym4', method='BayesShrink', mode='soft', wavelet_levels=5)

        # --- Peak Finding ---
        # A robust method to find significant peaks.
        min_prominence = (np.max(signal) - np.min(signal)) * 0.1 # Require prominence of at least 10% of signal range
        peaks, _ = find_peaks(signal, prominence=min_prominence)

        if len(peaks) < 3:
            raise ValueError("Fewer than 3 peaks were detected, cannot perform transient analysis.")
        
        # --- Calculate BPM and Peak-to-Peak Duration ---
        # Total duration of the signal in seconds for BPM calculation.
        signal_duration_seconds = len(signal) / frame_rate
        beats_per_minute = (len(peaks) / signal_duration_seconds) * 60 if signal_duration_seconds > 0 else 0
        
        # Mean time between peaks in milliseconds
        mean_peak_to_peak_duration = np.mean(np.diff(peaks)) * frame_interval if len(peaks) > 1 else None

        # --- Transient Analysis ---
        gradient1 = np.gradient(signal)
        gradient2 = np.gradient(gradient1)

        # Find the initiation point for each peak
        transient_defs = []
        for p in peaks:
            search_limit = p - (np.argmax(p > peaks) -1 if p > peaks[0] else 0) # Search back to previous peak
            init_time = -1
            for j in range(1, search_limit):
                idx = p - j
                if gradient1[idx] <= 0 and (idx + 1 < len(gradient2)) and gradient2[idx + 1] >= 0:
                    init_time = idx + 1
                    break
            if init_time != -1:
                transient_defs.append({'peak': p, 'initiation': init_time})

        if len(transient_defs) < 2:
            raise ValueError("Could not determine enough transient initiation points.")

        # Define full transients, where the end of one is the start of the next
        full_transients = []
        for i in range(len(transient_defs) - 1):
            current_transient = transient_defs[i]
            next_transient = transient_defs[i+1]
            full_transients.append({
                'initiation': current_transient['initiation'],
                'peak': current_transient['peak'],
                'end': next_transient['initiation']
            })
        
        if not full_transients:
            raise ValueError("No valid transients could be defined.")

        # --- Filter Transients based on 90% Decay Rule ---
        final_transients = []
        for t in full_transients:
            peak_val = signal[t['peak']]
            baseline_val = signal[t['initiation']]
            
            if peak_val <= baseline_val: continue
                
            decay_target = peak_val - (peak_val - baseline_val) * 0.9
            reaches_90_decay = any(signal[k] <= decay_target for k in range(t['peak'], t['end']))
            
            if reaches_90_decay:
                t['baseline'] = baseline_val
                t['peak_val'] = peak_val
                final_transients.append(t)
        
        if not final_transients:
            raise ValueError("No transients satisfied the 90% decay criteria.")

        # --- Calculate Parameters for each valid transient ---
        amplitudes, time_to_peaks, reach50s = [], [], []
        decay50s, decay80s, decay90s = [], [], []
        max_decay_rates, times_to_max_decay, rate_rsqs = [], [], []

        for t in final_transients:
            init, peak, end, baseline, peak_val = t['initiation'], t['peak'], t['end'], t['baseline'], t['peak_val']
            
            amplitudes.append(peak_val / baseline)
            time_to_peaks.append((peak - init) * frame_interval)
            
            # Rise time (R50)
            r50_target = baseline + (peak_val - baseline) * 0.5
            for k in range(init, peak):
                if signal[k] >= r50_target:
                    reach50s.append((k - init) * frame_interval)
                    break
            
            # Decay Times (T50, T80, T90)
            d50_target, d80_target, d90_target = [peak_val - (peak_val - baseline) * p for p in [0.5, 0.8, 0.9]]
            d50_found, d80_found, d90_found = False, False, False
            for k in range(peak, end):
                if not d50_found and signal[k] <= d50_target:
                    decay50s.append((k - peak) * frame_interval); d50_found = True
                if not d80_found and signal[k] <= d80_target:
                    decay80s.append((k - peak) * frame_interval); d80_found = True
                if not d90_found and signal[k] <= d90_target:
                    decay90s.append((k - peak) * frame_interval); d90_found = True
                    break

            # Max Decay Rate
            d90_frames = int(decay90s[-1] / frame_interval) if d90_found else (end - peak)
            decay_end_idx = peak + d90_frames
            if decay_end_idx > peak + 1 and decay_end_idx < len(signal):
                decay_signal = signal[peak:decay_end_idx]
                time_vector = np.arange(len(decay_signal)) * frame_interval
                
                poly_coeffs = np.polyfit(time_vector, decay_signal, 3)
                poly_der = np.polyder(poly_coeffs)
                
                fine_time = np.linspace(time_vector[0], time_vector[-1], num=len(time_vector)*10)
                der_values = np.polyval(poly_der, fine_time)
                
                max_decay_rates.append(abs(np.min(der_values)))
                times_to_max_decay.append(fine_time[np.argmin(der_values)])
                
                y_fit = np.polyval(poly_coeffs, time_vector)
                ss_res = np.sum((decay_signal - y_fit) ** 2)
                ss_tot = np.sum((decay_signal - np.mean(decay_signal)) ** 2)
                rate_rsqs.append(1 - (ss_res / ss_tot) if ss_tot > 0 else 1)

        # --- Generate Activation Map ---
        velocity_x, velocity_y, velocity_magnitude, velocity_r_squared = 0.0, 0.0, 0.0, 0.0
        try:
            print("Generating activation map...")
            activation_map, mapping_mask, signal_max, signal_min, signal_range, velocity_x, velocity_y, velocity_magnitude, velocity_r_squared = generate_activation_map(
                video_qc=video_qc,
                peaks=peaks,
                mean_peak_to_peak_duration=mean_peak_to_peak_duration,
                frame_interval=frame_interval,
                results_path=results_path,
                filename_only=filename_only,
                f=50000  # Default number of grids
            )
            print("Activation map generated successfully.")
        except Exception as e:
            print(f"Warning: Could not generate activation map: {str(e)}")
            # Continue with the analysis even if activation map fails

        # --- Summarize Results for this one video ---
        summary_dict = {
            "video_filename": filename_only,
            "beats_per_minute": beats_per_minute,
            "peak_to_peak_duration_ms": mean_peak_to_peak_duration,
            "amplitude_ratio_peak_baseline": np.mean(amplitudes) if amplitudes else None,
            "time_to_peak_ms": np.mean(time_to_peaks) if time_to_peaks else None,
            "time_to_50_rise_ms": np.mean(reach50s) if reach50s else None,
            "time_to_50_decay_ms": np.mean(decay50s) if decay50s else None,
            "time_to_80_decay_ms": np.mean(decay80s) if decay80s else None,
            "time_to_90_decay_ms": np.mean(decay90s) if decay90s else None,
            "time_to_max_decay_rate_ms": np.mean(times_to_max_decay) if times_to_max_decay else None,
            "max_decay_rate": np.mean(max_decay_rates) if max_decay_rates else None,
            "decay_fit_r_squared": np.mean(rate_rsqs) if rate_rsqs else None,
            "velocity_magnitude_pixels_per_ms": velocity_magnitude,
            "velocity_x_pixels_per_ms": velocity_x,
            "velocity_y_pixels_per_ms": velocity_y,
            "velocity_r_squared": velocity_r_squared
        }
        
        # Append to session results
        session['results_data'].append(summary_dict)

        # --- Generate and Save Plot ---
        time_axis = np.arange(len(signal)) * frame_interval
        
        # Create plot with raw/denoised signals and initiation points
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2,1,1)
        plt.plot(time_axis, raw_signal, alpha=0.6, label='Raw Signal')
        plt.plot(time_axis, signal, label='Denoised Signal')
        plt.plot(time_axis[peaks], signal[peaks], "x", label="Detected Peaks")
        plt.title(f"Signal for {filename_only}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Intensity")
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot(time_axis, signal, label='Denoised Signal')
        for t in final_transients:
            plt.axvline(x=time_axis[t['initiation']], color='r', linestyle='--', alpha=0.8)
        plt.title("Detected Transient Initiations")
        plt.xlabel("Time (ms)")
        plt.ylabel("Intensity")
        plt.legend(['Denoised Signal', 'Initiation Points'])

        plot_path = results_path / f"{filename_only}_transients_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Determine next step
        next_file_index = file_index + 1
        if next_file_index >= len(session['files']):
            next_file_index = None # This was the last file
            # This was the last file, so save the combined CSV file.
            if session.get('results_data'):
                results_df = pd.DataFrame(session['results_data'])
                csv_path = results_path / "combined_results.csv"
                results_df.to_csv(csv_path, index=False)

        # Return success response
        return JSONResponse(content={
            "message": "Analysis completed successfully.",
            "batch_id": batch_id,
            "next_file_index": next_file_index
        })
    except Exception as e:
        # Log the error for debugging
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/results/{folder_name}")
async def results_page(folder_name: str):
    results_dir = Path("results") / folder_name
    if not results_dir.is_dir():
        return HTMLResponse("Results not found.", status_code=404)

    files = sorted([f.name for f in results_dir.iterdir()])

    # Find plots, activation maps, velocity files, and combined CSV file
    plot_files = [f for f in files if f.endswith("_transients_plot.png")]
    activation_files = [f for f in files if f.endswith("_activation_map.jpg")]
    velocity_files = [f for f in files if f.endswith("_velocity_analysis.csv")]
    csv_file_path = results_dir / "combined_results.csv"

    # Read combined CSV and convert summary to HTML table
    table_html = "<h3>Summary data file (combined_results.csv) not found.</h3>"
    if csv_file_path.exists():
        try:
            df = pd.read_csv(csv_file_path)
            df_display = df.rename(columns=COLUMN_MAPPING)
            table_html = df_display.to_html(index=False, classes='results-table')
        except Exception as e:
            table_html = f"<h3>Error reading summary table: {e}</h3>"

    # Clean up session data to free memory
    if folder_name in SESSIONS:
        # Also clean up uploaded files and frames to save space
        upload_dir = Path("uploads") / folder_name
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        static_dir = Path("static/frames") / folder_name
        if static_dir.exists():
            shutil.rmtree(static_dir)
        del SESSIONS[folder_name]


    return HTMLResponse(get_results_html(folder_name, files, plot_files, activation_files, velocity_files, table_html))


@app.get("/results/{folder_name}/{file_name}")
async def get_result_file(folder_name: str, file_name: str):
    file_path = Path("results") / folder_name / file_name
    if not file_path.exists():
        return HTMLResponse("File not found.", status_code=404)
    return FileResponse(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
