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
# skimage, openpyxl, and pywavelets are also required.
# You can install them with: pip install scikit-image openpyxl pywavelets
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt


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
        results_so_far_html = f"""
        <h2>Latest Result</h2>
        <p>Plot for the previously analyzed video:</p>
        <img src="{latest_plot_url}" alt="Latest transient plot" style="max-width: 80%; height: auto; border: 1px solid #ccc; padding: 5px;">
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


def get_results_html(folder_name: str, files: list, plot_files: list[str], table_html: str):
    list_items = "".join(f'<li><a href="/results/{folder_name}/{f}" target="_blank">{f}</a></li>' for f in files)

    plots_html_content = ""
    if plot_files:
        plots_html_content += "<h2>Transient Plots</h2>"
        for plot_file in plot_files:
            plot_url = f"/results/{folder_name}/{plot_file}"
            plots_html_content += f'<div><h3 style="margin-top: 1.5em;">{plot_file}</h3><img src="{plot_url}" alt="Transient plot: {plot_file}" style="max-width: 100%; height: auto;"></div>'


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
        all_frames = [] # For saving video later if needed
        
        # Skip first 5 frames
        for _ in range(5):
            cap.read()

        for t in range(frame_count_adjusted):
            ret, frame = cap.read()
            if not ret:
                break
            # No need to save all frames for video writing in this version
            # if save_video_flag:
            #     all_frames.append(frame)
            
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
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
            "decay_fit_r_squared": np.mean(rate_rsqs) if rate_rsqs else None
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

    # Find plots and combined CSV file
    plot_files = [f for f in files if f.endswith("_transients_plot.png")]
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


    return HTMLResponse(get_results_html(folder_name, files, plot_files, table_html))


@app.get("/results/{folder_name}/{file_name}")
async def get_result_file(folder_name: str, file_name: str):
    file_path = Path("results") / folder_name / file_name
    if not file_path.exists():
        return HTMLResponse("File not found.", status_code=404)
    return FileResponse(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
