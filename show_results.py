"""
Functions to visualize and display experiment results in Jupyter notebooks.
"""

import os
import json
import glob
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Audio, HTML
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
import hashlib


def find_latest_run(mode=None):
    """
    Find the latest run directory.
    
    Args:
        mode: Optional mode string (mode1, mode2, mode3). If None, finds latest across all modes.
        
    Returns:
        Tuple of (mode_dir, run_dir, full_path) or (None, None, None) if no runs found
    """
    if mode:
        mode_dirs = [mode]
    else:
        mode_dirs = ['mode1', 'mode2', 'mode3']
    
    latest_run = None
    latest_time = 0
    
    for mode_dir in mode_dirs:
        mode_path = os.path.join('.', mode_dir)
        if not os.path.exists(mode_path):
            continue
        
        # Find all run directories
        run_dirs = glob.glob(os.path.join(mode_path, 'run_*'))
        for run_dir in run_dirs:
            if os.path.isdir(run_dir):
                # Get modification time
                mtime = os.path.getmtime(run_dir)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_run = (mode_dir, os.path.basename(run_dir), run_dir)
    
    return latest_run if latest_run else (None, None, None)


def load_metadata(run_dir):
    """Load metadata JSON from run directory."""
    metadata_path = os.path.join(run_dir, 'inputs_and_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def check_clipping(audio_path, normalization_factor=None, threshold=1.0):
    """
    Check if audio has clipping (peak > threshold).
    
    Args:
        audio_path: Path to audio file
        normalization_factor: Optional normalization factor to check against
        threshold: Threshold for clipping detection (default 1.0)
        
    Returns:
        Tuple of (has_clipping, max_value, warning_message)
    """
    audio, sr = torchaudio.load(audio_path)
    audio_flat = audio.view(-1).numpy()
    max_value = float(np.abs(audio_flat).max())
    
    has_clipping = max_value > threshold
    warning = None
    
    if has_clipping:
        if normalization_factor:
            # Calculate what the original max would have been before normalization
            # If saved audio has max > 1, and we know the normalization factor,
            # then original_max = saved_max * normalization_factor
            original_max = max_value * normalization_factor
            warning = f"⚠️  WARNING: Audio has clipping (peak = {max_value:.4f} > {threshold:.1f}). "
            warning += f"Original max before saving was {original_max:.4f} (normalization factor: {normalization_factor:.6f}). "
            warning += f"This audio may have been clipped when saved as a WAV file!"
        else:
            warning = f"⚠️  WARNING: Audio has clipping (peak = {max_value:.4f} > {threshold:.1f})!"
    elif normalization_factor:
        # Even if not clipping, show what the original max was
        original_max = max_value * normalization_factor
        if original_max > threshold:
            warning = f"ℹ️  Note: Audio peak is {max_value:.4f} (within limit). Original max before saving was {original_max:.4f}."
    
    return has_clipping, max_value, warning


def get_default_params_denormalized():
    """
    Get default parameters in denormalized form.
    Returns dictionary with denormalized default values.
    """
    # Import here to avoid circular imports
    from deepafx_st.processors.autodiff.peq import ParametricEQ
    from deepafx_st.processors.autodiff.compressor import Compressor
    import torch
    
    # Default normalized parameters
    default_norm = {
        'low_shelf_gain': 0.5,
        'low_shelf_cutoff': 0.4444,
        'low_shelf_q': 0.5,
        'band1_gain': 0.5,
        'band1_cutoff': 0.1111,
        'band1_q': 0.5,
        'band2_gain': 0.5,
        'band2_cutoff': 0.2105,
        'band2_q': 0.0615,
        'band3_gain': 0.5,
        'band3_cutoff': 0.3333,
        'band3_q': 0.0615,
        'band4_gain': 0.5,
        'band4_cutoff': 0.5833,
        'band4_q': 0.0615,
        'high_shelf_gain': 0.5,
        'high_shelf_cutoff': 0.5833,
        'high_shelf_q': 0.0615,
        'threshold': 0.85,
        'ratio': 0.0,
        'attack': 0.009,
        'release': 0.4,
        'knee': 0.5,
        'makeup_gain': 0.5,
    }
    
    # Denormalize
    peq = ParametricEQ(24000)
    comp = Compressor(24000)
    
    peq_params_norm = torch.tensor([
        default_norm['low_shelf_gain'],
        default_norm['low_shelf_cutoff'],
        default_norm['low_shelf_q'],
        default_norm['band1_gain'],
        default_norm['band1_cutoff'],
        default_norm['band1_q'],
        default_norm['band2_gain'],
        default_norm['band2_cutoff'],
        default_norm['band2_q'],
        default_norm['band3_gain'],
        default_norm['band3_cutoff'],
        default_norm['band3_q'],
        default_norm['band4_gain'],
        default_norm['band4_cutoff'],
        default_norm['band4_q'],
        default_norm['high_shelf_gain'],
        default_norm['high_shelf_cutoff'],
        default_norm['high_shelf_q'],
    ])
    
    comp_params_norm = torch.tensor([
        default_norm['threshold'],
        default_norm['ratio'],
        default_norm['attack'],
        default_norm['release'],
        default_norm['knee'],
        default_norm['makeup_gain'],
    ])
    
    peq_params_denorm = peq.denormalize_params(peq_params_norm)
    comp_params_denorm = comp.denormalize_params(comp_params_norm)
    
    # Convert to dictionary
    default_denorm = {}
    peq_param_names = [
        'low_shelf_gain', 'low_shelf_cutoff', 'low_shelf_q',
        'band1_gain', 'band1_cutoff', 'band1_q',
        'band2_gain', 'band2_cutoff', 'band2_q',
        'band3_gain', 'band3_cutoff', 'band3_q',
        'band4_gain', 'band4_cutoff', 'band4_q',
        'high_shelf_gain', 'high_shelf_cutoff', 'high_shelf_q',
    ]
    comp_param_names = ['threshold', 'ratio', 'attack', 'release', 'knee', 'makeup_gain']
    
    for i, name in enumerate(peq_param_names):
        default_denorm[name] = float(peq_params_denorm[i])
    
    for i, name in enumerate(comp_param_names):
        default_denorm[name] = float(comp_params_denorm[i])
    
    return default_denorm


def create_parameter_image(params_dict, title="Parameters", compare_to_defaults=True, 
                          tolerance=0.01, width=2000, bg_color=(255, 248, 220)):
    """
    Create an image showing parameters with formatting.
    
    Args:
        params_dict: Dictionary of denormalized parameters
        title: Title for the parameter display
        compare_to_defaults: Whether to compare to defaults and highlight differences
        tolerance: Tolerance for considering values equal
        width: Image width in pixels (increased for better resolution)
        bg_color: Background color (RGB tuple)
        
    Returns:
        PIL Image object
    """
    if params_dict is None:
        # Return a simple "No parameters" image
        img = Image.new('RGB', (width, 150), bg_color)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
        draw.text((20, 60), "No parameters available", fill=(0, 0, 0), font=font)
        return img
    
    default_params = get_default_params_denormalized() if compare_to_defaults else {}
    
    # Calculate required height with larger spacing
    line_height = 45  # Increased significantly for better readability
    padding = 30  # Increased for better margins
    title_height = 50  # Increased for title spacing
    section_spacing = 20  # Increased for section separation
    peq_lines = 6  # bands
    comp_lines = 6  # compressor params
    total_height = (padding * 2 + title_height + section_spacing * 2 + 
                   peq_lines * line_height + comp_lines * line_height + 50)
    
    # Create image
    img = Image.new('RGB', (width, total_height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load fonts - handle different font paths with larger sizes
    title_font = ImageFont.load_default()
    header_font = ImageFont.load_default()
    text_font = ImageFont.load_default()
    
    # Try macOS fonts for titles/headers with larger sizes
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                title_font = ImageFont.truetype(font_path, 40)  # Large for high-res display
                header_font = ImageFont.truetype(font_path, 28)  # Large for high-res display
                break
        except:
            continue
    
    # Try monospace fonts for parameter values with larger sizes
    mono_paths = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/Library/Fonts/Courier New.ttf",
    ]
    
    for mono_path in mono_paths:
        try:
            if os.path.exists(mono_path):
                text_font = ImageFont.truetype(mono_path, 22)  # Large for high-res display
                break
        except:
            continue
    
    y_pos = padding
    
    # Title
    draw.text((padding, y_pos), title, fill=(0, 0, 0), font=title_font)
    y_pos += title_height
    
    # PEQ section
    draw.text((padding, y_pos), "PEQ:", fill=(0, 0, 0), font=header_font)
    y_pos += line_height
    
    bands = [
        ('low_shelf', 'Low Shelf'),
        ('band1', 'Band 1'),
        ('band2', 'Band 2'),
        ('band3', 'Band 3'),
        ('band4', 'Band 4'),
        ('high_shelf', 'High Shelf'),
    ]
    
    for band_key, band_name in bands:
        gain = params_dict.get(f'{band_key}_gain', 0)
        cutoff = params_dict.get(f'{band_key}_cutoff', 0)
        q = params_dict.get(f'{band_key}_q', 0)
        
        if compare_to_defaults:
            gain_diff = abs(gain - default_params.get(f'{band_key}_gain', 0)) > tolerance
            cutoff_diff = abs(cutoff - default_params.get(f'{band_key}_cutoff', 0)) > tolerance
            q_diff = abs(q - default_params.get(f'{band_key}_q', 0)) > tolerance
        else:
            gain_diff = False
            cutoff_diff = False
            q_diff = False
        
        # Build line text with tighter spacing
        x = padding + 30
        line_text = f"{band_name:18s}"
        draw.text((x, y_pos), line_text, fill=(0, 0, 0), font=text_font)
        x += 200  # Reduced spacing - tighter layout
        
        # Gain - draw twice with slight offset for bold effect if needed
        gain_text = f"G={gain:.2f}dB"
        text_color = (255, 0, 0) if gain_diff else (0, 0, 0)
        if gain_diff:
            # Draw twice with slight offset for bold effect
            draw.text((x+1, y_pos), gain_text, fill=text_color, font=text_font)
            draw.text((x, y_pos+1), gain_text, fill=text_color, font=text_font)
        draw.text((x, y_pos), gain_text, fill=text_color, font=text_font)
        x += 140  # Reduced spacing
        
        # Cutoff
        cutoff_text = f"C={cutoff:.0f}Hz"
        text_color = (255, 0, 0) if cutoff_diff else (0, 0, 0)
        if cutoff_diff:
            draw.text((x+1, y_pos), cutoff_text, fill=text_color, font=text_font)
            draw.text((x, y_pos+1), cutoff_text, fill=text_color, font=text_font)
        draw.text((x, y_pos), cutoff_text, fill=text_color, font=text_font)
        x += 150  # Reduced spacing
        
        # Q
        q_text = f"Q={q:.2f}"
        text_color = (255, 0, 0) if q_diff else (0, 0, 0)
        if q_diff:
            draw.text((x+1, y_pos), q_text, fill=text_color, font=text_font)
            draw.text((x, y_pos+1), q_text, fill=text_color, font=text_font)
        draw.text((x, y_pos), q_text, fill=text_color, font=text_font)
        
        y_pos += line_height
    
    y_pos += section_spacing
    
    # Compressor section
    draw.text((padding, y_pos), "Compressor:", fill=(0, 0, 0), font=header_font)
    y_pos += line_height
    
    comp_params = [
        ('threshold', 'Threshold', 'dB'),
        ('ratio', 'Ratio', ''),
        ('attack', 'Attack', 'ms'),
        ('release', 'Release', 'ms'),
        ('knee', 'Knee', 'dB'),
        ('makeup_gain', 'Makeup', 'dB'),
    ]
    
    for param_key, param_name, unit in comp_params:
        value = params_dict.get(param_key, 0)
        
        if compare_to_defaults:
            is_diff = abs(value - default_params.get(param_key, 0)) > tolerance
        else:
            is_diff = False
        
        # Convert to appropriate units
        if unit == 'ms' and param_key in ['attack', 'release']:
            display_value = value * 1000
        else:
            display_value = value
        
        # Build line with tighter spacing
        x = padding + 30
        name_text = f"{param_name:18s}"
        draw.text((x, y_pos), name_text, fill=(0, 0, 0), font=text_font)
        x += 200  # Reduced spacing - tighter layout
        
        value_text = f"{display_value:.2f} {unit}"
        text_color = (255, 0, 0) if is_diff else (0, 0, 0)
        if is_diff:
            # Draw twice with slight offset for bold effect
            draw.text((x+1, y_pos), value_text, fill=text_color, font=text_font)
            draw.text((x, y_pos+1), value_text, fill=text_color, font=text_font)
        draw.text((x, y_pos), value_text, fill=text_color, font=text_font)
        
        y_pos += line_height
    
    return img


def get_or_create_parameter_image(params_dict, run_dir, image_name, title="Parameters",
                                  compare_to_defaults=True):
    """
    Get parameter image from disk or create it if it doesn't exist.
    
    Args:
        params_dict: Dictionary of denormalized parameters
        run_dir: Directory to save/load image from
        image_name: Filename for the image (e.g., 'params_actual.png')
        title: Title for the parameter display
        compare_to_defaults: Whether to compare to defaults
        
    Returns:
        Path to the parameter image
    """
    image_path = os.path.join(run_dir, image_name)
    
    # Check if image already exists
    if os.path.exists(image_path):
        return image_path
    
    # Create image
    img = create_parameter_image(params_dict, title, compare_to_defaults)
    img.save(image_path)
    
    return image_path


def create_side_by_side_param_image(params_dict1, params_dict2, title1="Actual", title2="Predicted",
                                    run_dir=None, image_name=None, compare_to_defaults=True):
    """
    Create a side-by-side comparison image of two parameter sets.
    
    Args:
        params_dict1: First parameter dictionary (actual)
        params_dict2: Second parameter dictionary (predicted)
        title1: Title for first parameter set
        title2: Title for second parameter set
        run_dir: Directory to save image to
        image_name: Filename for the image
        compare_to_defaults: Whether to compare to defaults
        
    Returns:
        Path to the image
    """
    width_per_panel = 2000  # High resolution for clarity
    spacing = 50  # Spacing between panels
    total_width = width_per_panel * 2 + spacing
    
    # Create both images
    img1 = create_parameter_image(params_dict1, title1, compare_to_defaults, width=width_per_panel)
    img2 = create_parameter_image(params_dict2, title2, compare_to_defaults, width=width_per_panel)
    
    # Use the taller image height
    height = max(img1.height, img2.height)
    
    # Create combined image
    combined_img = Image.new('RGB', (total_width, height), (255, 248, 220))
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (width_per_panel + spacing, 0))
    
    # Save if path provided
    if run_dir and image_name:
        image_path = os.path.join(run_dir, image_name)
        combined_img.save(image_path)
        return image_path
    
    return combined_img


def plot_spectrogram_with_params(audio_path, title="Spectrogram", params_dict=None, 
                                  params_dict2=None, run_dir=None, model_type=None,
                                  compare_to_defaults=True, sample_rate=24000):
    """
    Plot spectrogram. All spectrograms have consistent dimensions.
    
    Args:
        audio_path: Path to audio file
        title: Title for the plot
        params_dict: Not used (kept for compatibility)
        params_dict2: Not used (kept for compatibility)
        run_dir: Not used (kept for compatibility)
        model_type: Not used (kept for compatibility)
        compare_to_defaults: Not used (kept for compatibility)
        sample_rate: Sample rate of audio
    """
    audio, sr = torchaudio.load(audio_path)
    audio_np = audio[0].numpy()
    
    # Compute spectrogram with consistent parameters
    n_fft = 2048
    hop_length = 512
    
    stft = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        power=2.0
    )(audio)
    
    spectrogram_db = 10 * np.log10(stft[0].numpy() + 1e-10)
    
    # Fixed spectrogram dimensions for consistency - ALL spectrograms use these
    fixed_duration = 5.0  # seconds
    fixed_freq_max = 12000  # Hz
    fixed_fig_height = 5  # Consistent height for all spectrograms
    
    # Create figure with consistent size
    fig, ax = plt.subplots(figsize=(14, fixed_fig_height))
    
    # Plot spectrogram
    im = ax.imshow(
        spectrogram_db,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, len(audio_np) / sr, 0, sr / 2],
        interpolation='nearest'
    )
    
    # Set consistent limits - ALL spectrograms use same limits for alignment
    ax.set_xlim(0, fixed_duration)
    ax.set_ylim(0, fixed_freq_max)
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Power (dB)')
    
    plt.tight_layout()
    return ax


def format_parameters_horizontal(params_dict, params_dict2=None, compare_to_defaults=True):
    """
    Format parameters horizontally in a structured way.
    
    Args:
        params_dict: Dictionary of denormalized parameters
        params_dict2: Optional second dictionary for comparison
        compare_to_defaults: Whether to highlight differences with bold
        
    Returns:
        Formatted text string with formatting markers for bold
    """
    if params_dict is None and params_dict2 is None:
        return "No parameters available"
    
    default_params = get_default_params_denormalized() if compare_to_defaults else {}
    tolerance = 0.01
    
    lines = []
    
    if params_dict and params_dict2:
        # Side-by-side comparison format - more structured
        lines.append("Actual Parameters" + " " * 70 + "Predicted Parameters")
        lines.append("=" * 160)
        lines.append("")
        
        # PEQ section with proper alignment
        lines.append("PEQ:")
        bands = [
            ('low_shelf', 'Low Shelf'),
            ('band1', 'Band 1'),
            ('band2', 'Band 2'),
            ('band3', 'Band 3'),
            ('band4', 'Band 4'),
            ('high_shelf', 'High Shelf'),
        ]
        
        for band_key, band_name in bands:
            # Actual values
            gain1 = params_dict.get(f'{band_key}_gain', 0)
            cutoff1 = params_dict.get(f'{band_key}_cutoff', 0)
            q1 = params_dict.get(f'{band_key}_q', 0)
            
            # Predicted values
            gain2 = params_dict2.get(f'{band_key}_gain', 0)
            cutoff2 = params_dict2.get(f'{band_key}_cutoff', 0)
            q2 = params_dict2.get(f'{band_key}_q', 0)
            
            # Check if different from defaults for bold formatting
            if compare_to_defaults:
                gain1_diff = abs(gain1 - default_params.get(f'{band_key}_gain', 0)) > tolerance
                cutoff1_diff = abs(cutoff1 - default_params.get(f'{band_key}_cutoff', 0)) > tolerance
                q1_diff = abs(q1 - default_params.get(f'{band_key}_q', 0)) > tolerance
                gain2_diff = abs(gain2 - default_params.get(f'{band_key}_gain', 0)) > tolerance
                cutoff2_diff = abs(cutoff2 - default_params.get(f'{band_key}_cutoff', 0)) > tolerance
                q2_diff = abs(q2 - default_params.get(f'{band_key}_q', 0)) > tolerance
            else:
                gain1_diff = cutoff1_diff = q1_diff = gain2_diff = cutoff2_diff = q2_diff = False
            
            # Format with bold markers
            gain1_str = f"**{gain1:6.2f}**" if gain1_diff else f"{gain1:6.2f}"
            cutoff1_str = f"**{cutoff1:6.0f}**" if cutoff1_diff else f"{cutoff1:6.0f}"
            q1_str = f"**{q1:4.2f}**" if q1_diff else f"{q1:4.2f}"
            gain2_str = f"**{gain2:6.2f}**" if gain2_diff else f"{gain2:6.2f}"
            cutoff2_str = f"**{cutoff2:6.0f}**" if cutoff2_diff else f"{cutoff2:6.0f}"
            q2_str = f"**{q2:4.2f}**" if q2_diff else f"{q2:4.2f}"
            
            line1 = f"  {band_name:12s} G={gain1_str}dB C={cutoff1_str}Hz Q={q1_str}"
            line2 = f"  {band_name:12s} G={gain2_str}dB C={cutoff2_str}Hz Q={q2_str}"
            lines.append(line1 + " " * 25 + line2)
        
        lines.append("")
        lines.append("Compressor:")
        # Align threshold with bands, others below
        comp_params = [
            ('threshold', 'Threshold', 'dB', True),  # Align with bands
            ('ratio', 'Ratio', '', False),  # Align with other compressor params
            ('attack', 'Attack', 'ms', False),
            ('release', 'Release', 'ms', False),
            ('knee', 'Knee', 'dB', False),
            ('makeup_gain', 'Makeup', 'dB', False),
        ]
        
        for param_key, param_name, unit, align_with_bands in comp_params:
            val1 = params_dict.get(param_key, 0)
            val2 = params_dict2.get(param_key, 0)
            
            if unit == 'ms' and param_key in ['attack', 'release']:
                display_val1 = val1 * 1000
                display_val2 = val2 * 1000
            else:
                display_val1 = val1
                display_val2 = val2
            
            # Check if different from defaults
            if compare_to_defaults:
                val1_diff = abs(val1 - default_params.get(param_key, 0)) > tolerance
                val2_diff = abs(val2 - default_params.get(param_key, 0)) > tolerance
            else:
                val1_diff = val2_diff = False
            
            # Format with bold markers
            val1_str = f"**{display_val1:8.2f}**" if val1_diff else f"{display_val1:8.2f}"
            val2_str = f"**{display_val2:8.2f}**" if val2_diff else f"{display_val2:8.2f}"
            
            if align_with_bands:
                # Threshold aligns with PEQ bands (same indentation as bands)
                line1 = f"  {param_name:12s} {val1_str} {unit}"
                line2 = f"  {param_name:12s} {val2_str} {unit}"
            else:
                # Other compressor params align with each other
                line1 = f"    {param_name:10s} {val1_str} {unit}"
                line2 = f"    {param_name:10s} {val2_str} {unit}"
            
            lines.append(line1 + " " * 25 + line2)
    
    elif params_dict:
        # Single parameter set - more structured format
        lines.append("Parameters:")
        lines.append("")
        
        # PEQ - format each band on separate line for readability
        lines.append("PEQ:")
        bands = [
            ('low_shelf', 'Low Shelf'),
            ('band1', 'Band 1'),
            ('band2', 'Band 2'),
            ('band3', 'Band 3'),
            ('band4', 'Band 4'),
            ('high_shelf', 'High Shelf'),
        ]
        
        for band_key, band_name in bands:
            gain = params_dict.get(f'{band_key}_gain', 0)
            cutoff = params_dict.get(f'{band_key}_cutoff', 0)
            q = params_dict.get(f'{band_key}_q', 0)
            
            # Check if different from defaults
            if compare_to_defaults:
                gain_diff = abs(gain - default_params.get(f'{band_key}_gain', 0)) > tolerance
                cutoff_diff = abs(cutoff - default_params.get(f'{band_key}_cutoff', 0)) > tolerance
                q_diff = abs(q - default_params.get(f'{band_key}_q', 0)) > tolerance
            else:
                gain_diff = cutoff_diff = q_diff = False
            
            # Format with bold markers
            gain_str = f"**{gain:6.2f}**" if gain_diff else f"{gain:6.2f}"
            cutoff_str = f"**{cutoff:6.0f}**" if cutoff_diff else f"{cutoff:6.0f}"
            q_str = f"**{q:4.2f}**" if q_diff else f"{q:4.2f}"
            
            lines.append(f"  {band_name:12s} G={gain_str}dB C={cutoff_str}Hz Q={q_str}")
        
        lines.append("")
        lines.append("Compressor:")
        comp_params = [
            ('threshold', 'Threshold', 'dB'),
            ('ratio', 'Ratio', ''),
            ('attack', 'Attack', 'ms'),
            ('release', 'Release', 'ms'),
            ('knee', 'Knee', 'dB'),
            ('makeup_gain', 'Makeup', 'dB'),
        ]
        
        for param_key, param_name, unit in comp_params:
            value = params_dict.get(param_key, 0)
            if unit == 'ms' and param_key in ['attack', 'release']:
                display_value = value * 1000
            else:
                display_value = value
            
            # Check if different from defaults
            if compare_to_defaults:
                is_diff = abs(value - default_params.get(param_key, 0)) > tolerance
            else:
                is_diff = False
            
            # Format with bold marker
            val_str = f"**{display_value:8.2f}**" if is_diff else f"{display_value:8.2f}"
            lines.append(f"  {param_name:12s} {val_str} {unit}")
    
    return "\n".join(lines)


def print_parameter_comparison(actual_params_dict, predicted_params_dict, model_name="Model"):
    """
    Print a clear comparison table of actual vs predicted parameters.
    
    Args:
        actual_params_dict: Dictionary of actual denormalized parameters
        predicted_params_dict: Dictionary of predicted denormalized parameters
        model_name: Name of the model (e.g., "Speech Model", "Music Model")
    """
    print("\n" + "="*100)
    print(f"{model_name} - Parameter Comparison: Actual vs Predicted")
    print("="*100)
    print(f"{'Parameter':<35} {'Actual':<25} {'Predicted':<25} {'Error':<15}")
    print("-"*100)
    
    # PEQ parameters
    print("\nPEQ Parameters:")
    bands = [
        ('low_shelf', 'Low Shelf'),
        ('band1', 'Band 1'),
        ('band2', 'Band 2'),
        ('band3', 'Band 3'),
        ('band4', 'Band 4'),
        ('high_shelf', 'High Shelf'),
    ]
    
    for band_key, band_name in bands:
        # Gain
        gain_actual = actual_params_dict.get(f'{band_key}_gain', 0)
        gain_pred = predicted_params_dict.get(f'{band_key}_gain', 0)
        gain_error = abs(gain_actual - gain_pred)
        print(f"  {band_name:15s} Gain  :     {gain_actual:>10.2f} dB        {gain_pred:>16.2f} dB        {gain_error:>14.2f} dB")
        
        # Cutoff
        cutoff_actual = actual_params_dict.get(f'{band_key}_cutoff', 0)
        cutoff_pred = predicted_params_dict.get(f'{band_key}_cutoff', 0)
        cutoff_error = abs(cutoff_actual - cutoff_pred)
        print(f"  {band_name:15s} Cutoff:   {cutoff_actual:>12.0f} Hz       {cutoff_pred:>17.0f} Hz       {cutoff_error:>15.0f} Hz")
        
        # Q
        q_actual = actual_params_dict.get(f'{band_key}_q', 0)
        q_pred = predicted_params_dict.get(f'{band_key}_q', 0)
        q_error = abs(q_actual - q_pred)
        print(f"  {band_name:15s} Q     :        {q_actual:>10.2f}          {q_pred:>17.2f}          {q_error:>15.2f}")
        print()
    
    # Compressor parameters
    print("\nCompressor Parameters:")
    comp_params = [
        ('threshold', 'Threshold', 'dB'),
        ('ratio', 'Ratio', ''),
        ('attack', 'Attack', 'ms'),
        ('release', 'Release', 'ms'),
        ('knee', 'Knee', 'dB'),
        ('makeup_gain', 'Makeup Gain', 'dB'),
    ]
    
    for param_key, param_name, unit in comp_params:
        val_actual = actual_params_dict.get(param_key, 0)
        val_pred = predicted_params_dict.get(param_key, 0)
        
        # Convert to appropriate units
        if unit == 'ms' and param_key in ['attack', 'release']:
            val_actual_display = val_actual * 1000
            val_pred_display = val_pred * 1000
        else:
            val_actual_display = val_actual
            val_pred_display = val_pred
        
        error = abs(val_actual_display - val_pred_display)
        
        if unit:
            print(f"  {param_name:15s}           {val_actual_display:>12.2f} {unit:<4}    {val_pred_display:>18.2f} {unit:<4}    {error:>16.2f} {unit:<4}")
        else:
            print(f"  {param_name:15s}           {val_actual_display:>15.2f}          {val_pred_display:>17.2f}          {error:>15.2f}")
    
    print("="*100 + "\n")


def display_audio_with_info(audio_path, title, normalization_factor=None):
    """
    Display audio player with title and clipping warning.
    
    Args:
        audio_path: Path to audio file
        title: Title for the audio display
        normalization_factor: Optional normalization factor for clipping check
    """
    if not os.path.exists(audio_path):
        display(HTML(f"<p style='color:red;'>{title}: File not found ({audio_path})</p>"))
        return
    
    # Check for clipping
    has_clipping, max_value, warning = check_clipping(audio_path, normalization_factor)
    
    # Display warning if clipping detected
    if warning:
        display(HTML(f"<p style='color:orange; font-weight:bold;'>{warning}</p>"))
    
    # Display title
    display(HTML(f"<h4>{title}</h4>"))
    display(HTML(f"<p>Peak value: {max_value:.6f}</p>"))
    
    # Display audio player
    display(Audio(audio_path, autoplay=False))




def show_results(mode=None, run_num=None):
    """
    Main function to display experiment results.
    
    Args:
        mode: Mode string (mode1, mode2, mode3). If None, uses latest run.
        run_num: Run number (e.g., 1, 2). If None, uses latest run.
    """
    # Find run directory
    if mode and run_num:
        run_dir = os.path.join('.', mode, f'run_{run_num}')
        if not os.path.exists(run_dir):
            display(HTML(f"<p style='color:red;'>Error: Run directory not found: {run_dir}</p>"))
            return
    else:
        # Find latest run
        mode_dir, run_name, run_dir = find_latest_run(mode)
        if not run_dir:
            display(HTML("<p style='color:red;'>Error: No runs found!</p>"))
            return
        mode = mode_dir
        run_num = run_name.replace('run_', '')
        display(HTML(f"<h3>📊 Displaying Results: {mode} / {run_name}</h3>"))
    
    # Load metadata
    try:
        metadata = load_metadata(run_dir)
    except Exception as e:
        display(HTML(f"<p style='color:red;'>Error loading metadata: {str(e)}</p>"))
        return
    
    # Display experiment info
    display(HTML(f"<h4>Experiment Information</h4>"))
    display(HTML(f"<ul>"))
    display(HTML(f"<li><b>Mode:</b> {metadata['mode']}</li>"))
    display(HTML(f"<li><b>X Input:</b> {metadata.get('x_input_filename', 'N/A')}</li>"))
    if 'y_input_filename' in metadata:
        display(HTML(f"<li><b>Y Input:</b> {metadata['y_input_filename']}</li>"))
    display(HTML(f"</ul>"))
    
    # Get normalization factors
    norm_factors = metadata.get('normalization_factors', {})
    
    # Mode-specific displays
    if metadata['mode'] == 'mode1':
        display(HTML("<hr><h3>Mode 1: Parameter Restoration</h3>"))
        
        # Step 1: Input audio
        display(HTML("<h4>Step 1: Input Audio (X_input)</h4>"))
        x_input_path = os.path.join(run_dir, metadata['x_input_file'])
        display_audio_with_info(x_input_path, "X Input", None)
        
        # Input has default parameters (no params to show)
        plot_spectrogram_with_params(x_input_path, "X Input Spectrogram", params_dict=None)
        plt.show()
        
        # Step 2: Processed audio
        display(HTML("<h4>Step 2: Processed Audio (X_processed)</h4>"))
        x_processed_path = os.path.join(run_dir, metadata['x_processed_file'])
        norm_factor = norm_factors.get('x_processed')
        display_audio_with_info(x_processed_path, "X Processed", norm_factor)
        
        plot_spectrogram_with_params(x_processed_path, "X Processed Spectrogram")
        plt.show()
        
        # Step 3: Model predictions
        display(HTML("<h4>Step 3: Model Predictions</h4>"))
        
        # Speech model
        if 'speech' in metadata.get('models_used', {}):
            display(HTML("<h5>Speech Model</h5>"))
            pred_speech_path = os.path.join(run_dir, metadata['models_used']['speech']['output_file'])
            norm_factor = norm_factors.get('x_predicted_speech')
            display_audio_with_info(pred_speech_path, "X Predicted (Speech)", norm_factor)
            
            plot_spectrogram_with_params(pred_speech_path, "X Predicted (Speech) Spectrogram")
            plt.show()
        
        # Print parameter comparisons
        gt_params = metadata.get('parameters_denormalized')
        if gt_params:
            print("\n" + "="*100)
            print("PARAMETER COMPARISONS")
            print("="*100)
            
            if 'speech' in metadata.get('models_used', {}):
                speech_params = metadata['models_used']['speech'].get('p_predicted_denormalized')
                if speech_params:
                    print_parameter_comparison(gt_params, speech_params, "Speech Model")
            
            # if 'music' in metadata.get('models_used', {}):
            #     music_params = metadata['models_used']['music'].get('p_predicted_denormalized')
            #     if music_params:
            #         print_parameter_comparison(gt_params, music_params, "Music Model")
        
        # Music model
        if 'music' in metadata.get('models_used', {}):
            display(HTML("<h5>Music Model</h5>"))
            pred_music_path = os.path.join(run_dir, metadata['models_used']['music']['output_file'])
            norm_factor = norm_factors.get('x_predicted_music')
            display_audio_with_info(pred_music_path, "X Predicted (Music)", norm_factor)
            
            plot_spectrogram_with_params(pred_music_path, "X Predicted (Music) Spectrogram")
            plt.show()
        
        # Print parameter comparisons
        gt_params = metadata.get('parameters_denormalized')
        if gt_params:
            print("\n" + "="*100)
            print("PARAMETER COMPARISONS")
            print("="*100)
            
            # if 'speech' in metadata.get('models_used', {}):
            #     speech_params = metadata['models_used']['speech'].get('p_predicted_denormalized')
            #     if speech_params:
            #         print_parameter_comparison(gt_params, speech_params, "Speech Model")
            
            if 'music' in metadata.get('models_used', {}):
                music_params = metadata['models_used']['music'].get('p_predicted_denormalized')
                if music_params:
                    print_parameter_comparison(gt_params, music_params, "Music Model")
    
    elif metadata['mode'] == 'mode2':
        display(HTML("<hr><h3>Mode 2: Cross-Audio Style Transfer with Known Parameters</h3>"))
        
        # Step 1: Input audios
        display(HTML("<h4>Step 1: Input Audios</h4>"))
        x_input_path = os.path.join(run_dir, metadata['x_input_file'])
        display_audio_with_info(x_input_path, "X Input", None)
        y_input_path = os.path.join(run_dir, metadata['y_input_file'])
        display_audio_with_info(y_input_path, "Y Input", None)
        
        # Inputs have default parameters (no params to show)
        plot_spectrogram_with_params(x_input_path, "X Input Spectrogram", params_dict=None)
        plt.show()
        plot_spectrogram_with_params(y_input_path, "Y Input Spectrogram", params_dict=None)
        plt.show()
        
        # Step 2: Processed Y audio
        display(HTML("<h4>Step 2: Processed Y Audio (Y_processed)</h4>"))
        y_processed_path = os.path.join(run_dir, metadata['y_processed_file'])
        norm_factor = norm_factors.get('y_processed')
        display_audio_with_info(y_processed_path, "Y Processed", norm_factor)
        
        plot_spectrogram_with_params(y_processed_path, "Y Processed Spectrogram")
        plt.show()
        
        # Step 3: Model predictions
        display(HTML("<h4>Step 3: Model Predictions</h4>"))
        
        if 'speech' in metadata.get('models_used', {}):
            display(HTML("<h5>Speech Model</h5>"))
            pred_speech_path = os.path.join(run_dir, metadata['models_used']['speech']['output_file'])
            norm_factor = norm_factors.get('x_predicted_speech')
            display_audio_with_info(pred_speech_path, "X Predicted (Speech)", norm_factor)
            
            plot_spectrogram_with_params(pred_speech_path, "X Predicted (Speech) Spectrogram")
            plt.show()
        
        # Print parameter comparisons
        gt_params = metadata.get('parameters_denormalized')
        if gt_params:
            print("\n" + "="*100)
            print("PARAMETER COMPARISONS")
            print("="*100)
            
            if 'speech' in metadata.get('models_used', {}):
                speech_params = metadata['models_used']['speech'].get('p_predicted_denormalized')
                if speech_params:
                    print_parameter_comparison(gt_params, speech_params, "Speech Model")
            
            # if 'music' in metadata.get('models_used', {}):
            #     music_params = metadata['models_used']['music'].get('p_predicted_denormalized')
            #     if music_params:
            #         print_parameter_comparison(gt_params, music_params, "Music Model")
        
        if 'music' in metadata.get('models_used', {}):
            display(HTML("<h5>Music Model</h5>"))
            pred_music_path = os.path.join(run_dir, metadata['models_used']['music']['output_file'])
            norm_factor = norm_factors.get('x_predicted_music')
            display_audio_with_info(pred_music_path, "X Predicted (Music)", norm_factor)
            
            plot_spectrogram_with_params(pred_music_path, "X Predicted (Music) Spectrogram")
            plt.show()
        
        # Print parameter comparisons
        gt_params = metadata.get('parameters_denormalized')
        if gt_params:
            print("\n" + "="*100)
            print("PARAMETER COMPARISONS")
            print("="*100)
            
            # if 'speech' in metadata.get('models_used', {}):
            #     speech_params = metadata['models_used']['speech'].get('p_predicted_denormalized')
            #     if speech_params:
            #         print_parameter_comparison(gt_params, speech_params, "Speech Model")
            
            if 'music' in metadata.get('models_used', {}):
                music_params = metadata['models_used']['music'].get('p_predicted_denormalized')
                if music_params:
                    print_parameter_comparison(gt_params, music_params, "Music Model")
    
    else:  # mode3
        display(HTML("<hr><h3>Mode 3: Automatic Style Transfer</h3>"))
        
        # Step 1: Input audios
        display(HTML("<h4>Step 1: Input Audios</h4>"))
        x_input_path = os.path.join(run_dir, metadata['x_input_file'])
        display_audio_with_info(x_input_path, "X Input", None)
        y_input_path = os.path.join(run_dir, metadata['y_input_file'])
        display_audio_with_info(y_input_path, "Y Input", None)
        
        # Inputs have default parameters (no params to show)
        plot_spectrogram_with_params(x_input_path, "X Input Spectrogram", params_dict=None)
        plt.show()
        plot_spectrogram_with_params(y_input_path, "Y Input Spectrogram", params_dict=None)
        plt.show()
        
        # Step 2: Model predictions
        display(HTML("<h4>Step 2: Model Predictions</h4>"))
        
        if 'speech' in metadata.get('models_used', {}):
            display(HTML("<h5>Speech Model</h5>"))
            pred_speech_path = os.path.join(run_dir, metadata['models_used']['speech']['output_file'])
            norm_factor = norm_factors.get('x_predicted_speech')
            display_audio_with_info(pred_speech_path, "X Predicted (Speech)", norm_factor)
            
            plot_spectrogram_with_params(pred_speech_path, "X Predicted (Speech) Spectrogram")
            plt.show()
        
        # Print parameter comparisons
        # gt_params = metadata.get('parameters_denormalized')
        #if gt_params:
        print("\n" + "="*100)
        print("PARAMETER COMPARISONS")
        print("="*100)
        
        if 'speech' in metadata.get('models_used', {}):
            speech_params = metadata['models_used']['speech'].get('p_predicted_denormalized')
            if speech_params:
                print_parameter_comparison(speech_params, speech_params, "Speech Model")
        
        # if 'music' in metadata.get('models_used', {}):
        #     music_params = metadata['models_used']['music'].get('p_predicted_denormalized')
        #     if music_params:
        #         print_parameter_comparison(music_params, music_params, "Music Model")
        
        if 'music' in metadata.get('models_used', {}):
            display(HTML("<h5>Music Model</h5>"))
            pred_music_path = os.path.join(run_dir, metadata['models_used']['music']['output_file'])
            norm_factor = norm_factors.get('x_predicted_music')
            display_audio_with_info(pred_music_path, "X Predicted (Music)", norm_factor)
            
            plot_spectrogram_with_params(pred_music_path, "X Predicted (Music) Spectrogram")
            plt.show()
        
        # Print parameter comparisons
        # gt_params = metadata.get('parameters_denormalized')
        # if gt_params:
        print("\n" + "="*100)
        print("PARAMETER COMPARISONS")
        print("="*100)
        
        # if 'speech' in metadata.get('models_used', {}):
        #     speech_params = metadata['models_used']['speech'].get('p_predicted_denormalized')
        #     if speech_params:
        #         print_parameter_comparison(gt_params, speech_params, "Speech Model")
        
        if 'music' in metadata.get('models_used', {}):
            music_params = metadata['models_used']['music'].get('p_predicted_denormalized')
            if music_params:
                print_parameter_comparison(music_params, music_params, "Music Model")


if __name__ == "__main__":
    # For testing
    show_results()

