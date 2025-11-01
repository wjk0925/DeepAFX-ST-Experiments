"""
Widget helpers for interactive notebook
This module contains helper functions to create interactive widgets for the notebook
"""

import os
import json
import subprocess
import ipywidgets as widgets
from IPython.display import display, clear_output
from deepafx_st.utils import denormalize


def create_mode_dropdown(options=['mode1', 'mode2', 'mode3'], default='mode1'):
    """
    Create a dropdown widget for mode selection.
    
    Args:
        options: List of mode options to display
        default: Default selected value
        
    Returns:
        mode_dropdown: The dropdown widget
        mode_output: Output widget that displays the selected mode
    """
    # Create a dropdown widget for mode selection
    mode_dropdown = widgets.Dropdown(
        options=options,
        value=default,
        description='Mode:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px', margin='10px 0px')
    )
    
    # Create an output widget to display the selected value dynamically
    mode_output = widgets.Output(layout=widgets.Layout(margin='10px 0px'))
    
    def update_mode_display(change):
        """Update display when mode changes"""
        with mode_output:
            mode_output.clear_output(wait=True)
            print(f"✓ Selected mode: {change['new']}")
    
    # Attach observer to update display when dropdown changes
    mode_dropdown.observe(update_mode_display, names='value')
    
    # Initial display
    with mode_output:
        print(f"✓ Selected mode: {mode_dropdown.value}")
    
    # Display the dropdown and output together
    display(widgets.VBox([
        widgets.HTML("<b>Select a mode:</b>"),
        mode_dropdown,
        mode_output
    ]))
    
    return mode_dropdown, mode_output


def get_audio_files(audio_dir='audios'):
    """
    Get list of audio files from the audios directory.
    
    Args:
        audio_dir: Directory containing audio files
        
    Returns:
        List of audio filenames (without path)
    """
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    audio_files = []
    
    if os.path.exists(audio_dir):
        for filename in os.listdir(audio_dir):
            if any(filename.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(filename)
    
    return sorted(audio_files)


def create_audio_selection(mode_dropdown, audio_dir='audios'):
    """
    Create audio selection widgets based on the selected mode.
    
    Args:
        mode_dropdown: The mode dropdown widget to observe
        audio_dir: Directory containing audio files
        
    Returns:
        audio_widgets: Dictionary containing the audio selection widgets
        audio_output: Output widget that displays selected audios
    """
    # Get available audio files
    audio_files = get_audio_files(audio_dir)
    
    # Try to use Combobox (available in ipywidgets 8.0+), fallback to Dropdown
    if hasattr(widgets, 'Combobox'):
        WidgetClass = widgets.Combobox
        use_combobox = True
    else:
        WidgetClass = widgets.Dropdown
        use_combobox = False
    
    if not audio_files:
        # If no audio files found, create empty widgets
        widget_kwargs = {
            'options': [],
            'value': '',
            'description': 'X_input:',
            'disabled': True,
            'style': {'description_width': 'initial'},
            'layout': widgets.Layout(width='400px', margin='5px 0px')
        }
        if use_combobox:
            widget_kwargs.update({
                'placeholder': 'No audio files found in audios folder',
                'ensure_option': False
            })
        
        x_input_widget = WidgetClass(**widget_kwargs)
        
        widget_kwargs['description'] = 'Y_input:'
        y_input_widget = WidgetClass(**widget_kwargs)
    else:
        # Create X_input widget (used in all modes) - start with no default value
        if use_combobox:
            # Combobox can have empty string as default
            widget_kwargs = {
                'options': audio_files,
                'value': '',  # Start blank, no default selection
                'placeholder': 'Type to search or select from list',
                'ensure_option': False,
                'description': 'X_input:',
                'style': {'description_width': 'initial'},
                'layout': widgets.Layout(width='400px', margin='5px 0px')
            }
        else:
            # Dropdown needs a valid option or None, add empty option at start
            widget_kwargs = {
                'options': [''] + audio_files,  # Add empty option at beginning
                'value': '',  # Start with empty selection
                'description': 'X_input:',
                'style': {'description_width': 'initial'},
                'layout': widgets.Layout(width='400px', margin='5px 0px')
            }
        
        x_input_widget = WidgetClass(**widget_kwargs)
        
        # Create Y_input widget (used in mode2 and mode3) - start with no default value
        if use_combobox:
            widget_kwargs_y = {
                'options': audio_files,
                'value': '',  # Start blank, no default selection
                'placeholder': 'Type to search or select from list',
                'ensure_option': False,
                'description': 'Y_input:',
                'style': {'description_width': 'initial'},
                'layout': widgets.Layout(width='400px', margin='5px 0px')
            }
        else:
            # Dropdown needs a valid option or None, add empty option at start
            widget_kwargs_y = {
                'options': [''] + audio_files,  # Add empty option at beginning
                'value': '',  # Start with empty selection
                'description': 'Y_input:',
                'style': {'description_width': 'initial'},
                'layout': widgets.Layout(width='400px', margin='5px 0px')
            }
        y_input_widget = WidgetClass(**widget_kwargs_y)
    
    # Create output widget to display selections
    audio_output = widgets.Output(layout=widgets.Layout(margin='10px 0px'))
    
    def update_audio_display():
        """Update display based on current mode and selections"""
        mode = mode_dropdown.value
        with audio_output:
            audio_output.clear_output(wait=True)
            
            if mode == 'mode1':
                print(f"✓ X_input: {x_input_widget.value or '(not selected)'}")
            elif mode in ['mode2', 'mode3']:
                print(f"✓ X_input: {x_input_widget.value or '(not selected)'}")
                print(f"✓ Y_input: {y_input_widget.value or '(not selected)'}")
    
    # Show/hide Y_input widget based on mode
    def update_widget_visibility():
        """Show/hide Y_input widget based on current mode"""
        mode = mode_dropdown.value
        if mode in ['mode2', 'mode3']:
            y_input_widget.layout.visibility = 'visible'
            y_input_widget.layout.display = 'flex'  # Ensure it's displayed
            y_input_widget.disabled = False
        else:  # mode1
            y_input_widget.layout.visibility = 'hidden'
            y_input_widget.layout.display = 'none'  # Completely hide it
            y_input_widget.disabled = True
    
    def on_mode_change(change):
        """Handle mode changes - update visibility and display"""
        update_widget_visibility()
        update_audio_display()
    
    def on_audio_change(change):
        """Handle audio selection changes"""
        update_audio_display()
    
    # Observe mode changes
    mode_dropdown.observe(on_mode_change, names='value')
    
    # Observe audio changes
    x_input_widget.observe(on_audio_change, names='value')
    y_input_widget.observe(on_audio_change, names='value')
    
    # Set initial visibility and display
    update_widget_visibility()
    update_audio_display()
    
    # Function to refresh audio files list (defined after widgets are created)
    def refresh_audio_files():
        """Refresh the list of audio files from the directory"""
        nonlocal audio_files
        audio_files = get_audio_files(audio_dir)
        
        # Update widget options while preserving current selections
        x_current = x_input_widget.value
        y_current = y_input_widget.value
        
        if use_combobox:
            x_input_widget.options = audio_files
            y_input_widget.options = audio_files
        else:
            x_input_widget.options = [''] + audio_files
            y_input_widget.options = [''] + audio_files
        
        # Restore selections if they still exist in the new list
        if x_current and x_current in audio_files:
            x_input_widget.value = x_current
        else:
            x_input_widget.value = ''
            
        if y_current and y_current in audio_files:
            y_input_widget.value = y_current
        else:
            y_input_widget.value = ''
        
        # Update display
        update_audio_display()
    
    # Create refresh button
    refresh_button = widgets.Button(
        description='🔄 Refresh Audio List',
        button_style='info',
        tooltip='Click to reload audio files from the audios folder',
        layout=widgets.Layout(width='200px', margin='5px 0px')
    )
    
    def on_refresh_clicked(button):
        """Handle refresh button click"""
        refresh_audio_files()
        with audio_output:
            audio_output.clear_output(wait=True)
            print(f"✓ Refreshed! Found {len(audio_files)} audio file(s)")
            update_audio_display()
    
    refresh_button.on_click(on_refresh_clicked)
    
    # Create container with all widgets (Y_input visibility controlled separately)
    container_box = widgets.VBox([
        widgets.HBox([
            widgets.HTML("<b>Select audio files:</b>"),
            refresh_button
        ]),
        x_input_widget,
        y_input_widget,
        audio_output
    ])
    
    # Display the container
    display(container_box)
    
    return {
        'X_input': x_input_widget,
        'Y_input': y_input_widget,
        'output': audio_output
    }


def create_parameter_widgets(mode_dropdown, default_params=None):
    """
    Create parameter input widgets for PEQ and Compressor parameters.
    
    Args:
        mode_dropdown: The mode dropdown widget to observe
        default_params: Dictionary of default parameter values
        
    Returns:
        param_widgets: Dictionary containing all parameter widgets
        param_output: Output widget that displays current parameter values
        param_container: Container widget with all parameter controls
    """
    # Default parameters
    if default_params is None:
        default_params = {
            # PEQ - Low Shelf
            'low_shelf_gain': 0.5,      # 0.5 = 0 dB (no change)
            'low_shelf_cutoff': 0.4444,  
            'low_shelf_q': 0.5,
            
            # PEQ - Band 1
            'band1_gain': 0.5,          # 0.5 = 0 dB (no change)
            'band1_cutoff': 0.1111,
            'band1_q': 0.5,
            
            # PEQ - Band 2
            'band2_gain': 0.5,          # 0.5 = 0 dB (no change)
            'band2_cutoff': 0.2105,
            'band2_q': 0.0615,
            
            # PEQ - Band 3
            'band3_gain': 0.5,          # 0.5 = 0 dB (no change)
            'band3_cutoff': 0.3333,
            'band3_q': 0.0615,
            
            # PEQ - Band 4
            'band4_gain': 0.5,          # 0.5 = 0 dB (no change)
            'band4_cutoff': 0.5833,
            'band4_q': 0.0615,
            
            # PEQ - High Shelf
            'high_shelf_gain': 0.5,     # 0.5 = 0 dB (no change)
            'high_shelf_cutoff': 0.5833,
            'high_shelf_q': 0.0615,
            
            # Compressor
            'threshold': 0.85,          # -12 dB
            'ratio': 0.0,               # 0.0 = 1:1 (bypass)
            'attack': 0.009,
            'release': 0.4,             # dummy parameter
            'knee': 0.5,
            'makeup_gain': 0.5,         # 0.5 = 0 dB (no change)
        }
    
    # Create parameter widgets (all normalized 0-1 range)
    param_widgets = {}
    
    # Helper function to create a FloatSlider
    def create_slider(name, default, description, min_val=0.0, max_val=1.0, step=0.01):
        return widgets.FloatSlider(
            value=default,
            min=min_val,
            max=max_val,
            step=step,
            description=description,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px', margin='3px 0px')
        )
    
    # PEQ - Low Shelf
    param_widgets['low_shelf_gain'] = create_slider(
        'low_shelf_gain', default_params['low_shelf_gain'], 'Low Shelf Gain:'
    )
    param_widgets['low_shelf_cutoff'] = create_slider(
        'low_shelf_cutoff', default_params['low_shelf_cutoff'], 'Low Shelf Cutoff:'
    )
    param_widgets['low_shelf_q'] = create_slider(
        'low_shelf_q', default_params['low_shelf_q'], 'Low Shelf Q:'
    )
    
    # PEQ - Band 1
    param_widgets['band1_gain'] = create_slider(
        'band1_gain', default_params['band1_gain'], 'Band 1 Gain:'
    )
    param_widgets['band1_cutoff'] = create_slider(
        'band1_cutoff', default_params['band1_cutoff'], 'Band 1 Cutoff:'
    )
    param_widgets['band1_q'] = create_slider(
        'band1_q', default_params['band1_q'], 'Band 1 Q:'
    )
    
    # PEQ - Band 2
    param_widgets['band2_gain'] = create_slider(
        'band2_gain', default_params['band2_gain'], 'Band 2 Gain:'
    )
    param_widgets['band2_cutoff'] = create_slider(
        'band2_cutoff', default_params['band2_cutoff'], 'Band 2 Cutoff:'
    )
    param_widgets['band2_q'] = create_slider(
        'band2_q', default_params['band2_q'], 'Band 2 Q:'
    )
    
    # PEQ - Band 3
    param_widgets['band3_gain'] = create_slider(
        'band3_gain', default_params['band3_gain'], 'Band 3 Gain:'
    )
    param_widgets['band3_cutoff'] = create_slider(
        'band3_cutoff', default_params['band3_cutoff'], 'Band 3 Cutoff:'
    )
    param_widgets['band3_q'] = create_slider(
        'band3_q', default_params['band3_q'], 'Band 3 Q:'
    )
    
    # PEQ - Band 4
    param_widgets['band4_gain'] = create_slider(
        'band4_gain', default_params['band4_gain'], 'Band 4 Gain:'
    )
    param_widgets['band4_cutoff'] = create_slider(
        'band4_cutoff', default_params['band4_cutoff'], 'Band 4 Cutoff:'
    )
    param_widgets['band4_q'] = create_slider(
        'band4_q', default_params['band4_q'], 'Band 4 Q:'
    )
    
    # PEQ - High Shelf
    param_widgets['high_shelf_gain'] = create_slider(
        'high_shelf_gain', default_params['high_shelf_gain'], 'High Shelf Gain:'
    )
    param_widgets['high_shelf_cutoff'] = create_slider(
        'high_shelf_cutoff', default_params['high_shelf_cutoff'], 'High Shelf Cutoff:'
    )
    param_widgets['high_shelf_q'] = create_slider(
        'high_shelf_q', default_params['high_shelf_q'], 'High Shelf Q:'
    )
    
    # Compressor
    param_widgets['threshold'] = create_slider(
        'threshold', default_params['threshold'], 'Compressor Threshold:'
    )
    param_widgets['ratio'] = create_slider(
        'ratio', default_params['ratio'], 'Compressor Ratio:'
    )
    param_widgets['attack'] = create_slider(
        'attack', default_params['attack'], 'Compressor Attack:'
    )
    param_widgets['release'] = create_slider(
        'release', default_params['release'], 'Compressor Release (dummy):'
    )
    param_widgets['knee'] = create_slider(
        'knee', default_params['knee'], 'Compressor Knee:'
    )
    param_widgets['makeup_gain'] = create_slider(
        'makeup_gain', default_params['makeup_gain'], 'Compressor Makeup Gain:'
    )
    
    # Parameter port information for denormalization (from PEQ and Compressor classes)
    param_ports = {
        # PEQ - Low Shelf
        'low_shelf_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'low_shelf_cutoff': {'min': 20.0, 'max': 200.0, 'units': 'Hz'},
        'low_shelf_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - Band 1
        'band1_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'band1_cutoff': {'min': 200.0, 'max': 2000.0, 'units': 'Hz'},
        'band1_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - Band 2
        'band2_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'band2_cutoff': {'min': 200.0, 'max': 4000.0, 'units': 'Hz'},
        'band2_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - Band 3
        'band3_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'band3_cutoff': {'min': 2000.0, 'max': 8000.0, 'units': 'Hz'},
        'band3_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - Band 4
        'band4_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'band4_cutoff': {'min': 4000.0, 'max': 10800.0, 'units': 'Hz'},  # (24000 // 2) * 0.9
        'band4_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - High Shelf
        'high_shelf_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'high_shelf_cutoff': {'min': 4000.0, 'max': 10800.0, 'units': 'Hz'},
        'high_shelf_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # Compressor
        'threshold': {'min': -80.0, 'max': 0.0, 'units': 'dB'},
        'ratio': {'min': 1.0, 'max': 20.0, 'units': ''},
        'attack': {'min': 0.0001, 'max': 0.1, 'units': 's'},
        'release': {'min': 0.005, 'max': 1.0, 'units': 's'},
        'knee': {'min': 0.0, 'max': 12.0, 'units': 'dB'},
        'makeup_gain': {'min': -48.0, 'max': 48.0, 'units': 'dB'},
    }
    
    def denormalize_param(param_name, norm_value):
        """Denormalize a single parameter from [0,1] to actual range"""
        if param_name in param_ports:
            port = param_ports[param_name]
            denorm_value = denormalize(norm_value, port['max'], port['min'])
            return denorm_value, port['units']
        return norm_value, ''
    
    # Create output widget
    param_output = widgets.Output(layout=widgets.Layout(margin='10px 0px'))
    
    def update_param_display():
        """Update parameter display with both normalized and denormalized values"""
        with param_output:
            param_output.clear_output(wait=True)
            mode = mode_dropdown.value
            if mode in ['mode1', 'mode2']:
                print("Current parameter values (normalized [0,1] → actual value):")
                # Group display by sections
                print("\n📊 PEQ Parameters:")
                # Low Shelf
                gain_val = param_widgets['low_shelf_gain'].value
                gain_denorm, gain_units = denormalize_param('low_shelf_gain', gain_val)
                cutoff_val = param_widgets['low_shelf_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('low_shelf_cutoff', cutoff_val)
                q_val = param_widgets['low_shelf_q'].value
                q_denorm, q_units = denormalize_param('low_shelf_q', q_val)
                print(f"  Low Shelf: Gain={gain_val:.4f} → {gain_denorm:.2f} {gain_units}, "
                      f"Cutoff={cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}, "
                      f"Q={q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # Band 1
                gain_val = param_widgets['band1_gain'].value
                gain_denorm, gain_units = denormalize_param('band1_gain', gain_val)
                cutoff_val = param_widgets['band1_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('band1_cutoff', cutoff_val)
                q_val = param_widgets['band1_q'].value
                q_denorm, q_units = denormalize_param('band1_q', q_val)
                print(f"  Band 1: Gain={gain_val:.4f} → {gain_denorm:.2f} {gain_units}, "
                      f"Cutoff={cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}, "
                      f"Q={q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # Band 2
                gain_val = param_widgets['band2_gain'].value
                gain_denorm, gain_units = denormalize_param('band2_gain', gain_val)
                cutoff_val = param_widgets['band2_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('band2_cutoff', cutoff_val)
                q_val = param_widgets['band2_q'].value
                q_denorm, q_units = denormalize_param('band2_q', q_val)
                print(f"  Band 2: Gain={gain_val:.4f} → {gain_denorm:.2f} {gain_units}, "
                      f"Cutoff={cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}, "
                      f"Q={q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # Band 3
                gain_val = param_widgets['band3_gain'].value
                gain_denorm, gain_units = denormalize_param('band3_gain', gain_val)
                cutoff_val = param_widgets['band3_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('band3_cutoff', cutoff_val)
                q_val = param_widgets['band3_q'].value
                q_denorm, q_units = denormalize_param('band3_q', q_val)
                print(f"  Band 3: Gain={gain_val:.4f} → {gain_denorm:.2f} {gain_units}, "
                      f"Cutoff={cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}, "
                      f"Q={q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # Band 4
                gain_val = param_widgets['band4_gain'].value
                gain_denorm, gain_units = denormalize_param('band4_gain', gain_val)
                cutoff_val = param_widgets['band4_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('band4_cutoff', cutoff_val)
                q_val = param_widgets['band4_q'].value
                q_denorm, q_units = denormalize_param('band4_q', q_val)
                print(f"  Band 4: Gain={gain_val:.4f} → {gain_denorm:.2f} {gain_units}, "
                      f"Cutoff={cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}, "
                      f"Q={q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # High Shelf
                gain_val = param_widgets['high_shelf_gain'].value
                gain_denorm, gain_units = denormalize_param('high_shelf_gain', gain_val)
                cutoff_val = param_widgets['high_shelf_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('high_shelf_cutoff', cutoff_val)
                q_val = param_widgets['high_shelf_q'].value
                q_denorm, q_units = denormalize_param('high_shelf_q', q_val)
                print(f"  High Shelf: Gain={gain_val:.4f} → {gain_denorm:.2f} {gain_units}, "
                      f"Cutoff={cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}, "
                      f"Q={q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                print("\n🎚️ Compressor Parameters:")
                thresh_val = param_widgets['threshold'].value
                thresh_denorm, thresh_units = denormalize_param('threshold', thresh_val)
                ratio_val = param_widgets['ratio'].value
                ratio_denorm, ratio_units = denormalize_param('ratio', ratio_val)
                attack_val = param_widgets['attack'].value
                attack_denorm, attack_units = denormalize_param('attack', attack_val)
                release_val = param_widgets['release'].value
                release_denorm, release_units = denormalize_param('release', release_val)
                knee_val = param_widgets['knee'].value
                knee_denorm, knee_units = denormalize_param('knee', knee_val)
                makeup_val = param_widgets['makeup_gain'].value
                makeup_denorm, makeup_units = denormalize_param('makeup_gain', makeup_val)
                print(f"  Threshold={thresh_val:.4f} → {thresh_denorm:.2f} {thresh_units}, "
                      f"Ratio={ratio_val:.4f} → {ratio_denorm:.2f} {ratio_units}, "
                      f"Attack={attack_val:.4f} → {attack_denorm:.4f} {attack_units}, "
                      f"Release={release_val:.4f} → {release_denorm:.3f} {release_units}, "
                      f"Knee={knee_val:.4f} → {knee_denorm:.2f} {knee_units}, "
                      f"Makeup Gain={makeup_val:.4f} → {makeup_denorm:.2f} {makeup_units}")
    
    # Observe parameter changes
    for widget in param_widgets.values():
        widget.observe(lambda change: update_param_display(), names='value')
    
    # Show/hide parameter widgets based on mode
    def update_param_visibility():
        """Show/hide parameter widgets based on current mode"""
        mode = mode_dropdown.value
        if mode in ['mode1', 'mode2']:
            # Show all parameter widgets
            for widget in param_widgets.values():
                widget.layout.visibility = 'visible'
                widget.layout.display = 'flex'
                widget.disabled = False
        else:  # mode3
            # Hide all parameter widgets
            for widget in param_widgets.values():
                widget.layout.visibility = 'hidden'
                widget.layout.display = 'none'
                widget.disabled = True
    
    def on_mode_change(change):
        """Handle mode changes"""
        update_param_visibility()
        update_param_display()
    
    # Observe mode changes
    mode_dropdown.observe(on_mode_change, names='value')
    
    # Set initial visibility and display
    update_param_visibility()
    update_param_display()
    
    # Create organized container with accordion for collapsible bands
    peq_low_shelf = widgets.VBox([
        widgets.HTML("<b>Low Shelf</b>"),
        param_widgets['low_shelf_gain'],
        param_widgets['low_shelf_cutoff'],
        param_widgets['low_shelf_q'],
    ])
    
    peq_band1 = widgets.VBox([
        widgets.HTML("<b>Band 1</b>"),
        param_widgets['band1_gain'],
        param_widgets['band1_cutoff'],
        param_widgets['band1_q'],
    ])
    
    peq_band2 = widgets.VBox([
        widgets.HTML("<b>Band 2</b>"),
        param_widgets['band2_gain'],
        param_widgets['band2_cutoff'],
        param_widgets['band2_q'],
    ])
    
    peq_band3 = widgets.VBox([
        widgets.HTML("<b>Band 3</b>"),
        param_widgets['band3_gain'],
        param_widgets['band3_cutoff'],
        param_widgets['band3_q'],
    ])
    
    peq_band4 = widgets.VBox([
        widgets.HTML("<b>Band 4</b>"),
        param_widgets['band4_gain'],
        param_widgets['band4_cutoff'],
        param_widgets['band4_q'],
    ])
    
    peq_high_shelf = widgets.VBox([
        widgets.HTML("<b>High Shelf</b>"),
        param_widgets['high_shelf_gain'],
        param_widgets['high_shelf_cutoff'],
        param_widgets['high_shelf_q'],
    ])
    
    compressor_params = widgets.VBox([
        widgets.HTML("<b>Compressor</b>"),
        param_widgets['threshold'],
        param_widgets['ratio'],
        param_widgets['attack'],
        param_widgets['release'],
        param_widgets['knee'],
        param_widgets['makeup_gain'],
    ])
    
    # Use accordion for collapsible sections
    peq_accordion = widgets.Accordion(children=[
        peq_low_shelf,
        peq_band1,
        peq_band2,
        peq_band3,
        peq_band4,
        peq_high_shelf,
    ])
    peq_accordion.set_title(0, 'Low Shelf')
    peq_accordion.set_title(1, 'Band 1')
    peq_accordion.set_title(2, 'Band 2')
    peq_accordion.set_title(3, 'Band 3')
    peq_accordion.set_title(4, 'Band 4')
    peq_accordion.set_title(5, 'High Shelf')
    
    # Create main container
    param_container = widgets.VBox([
        widgets.HTML("<h3>📊 DSP Parameters</h3>"),
        widgets.HTML("<i>Only used in Mode 1 and Mode 2</i>"),
        widgets.HTML("<h4>PEQ (Parametric Equalizer)</h4>"),
        peq_accordion,
        widgets.HTML("<h4>Compressor</h4>"),
        compressor_params,
        param_output
    ])
    
    # Display the container
    display(param_container)
    
    return param_widgets, param_output, param_container


def create_review_and_run(mode_dropdown, audio_widgets, param_widgets, run_script='run.sh'):
    """
    Create a review block that displays all user inputs and a run button.
    
    Args:
        mode_dropdown: The mode dropdown widget
        audio_widgets: Dictionary containing audio selection widgets
        param_widgets: Dictionary containing parameter widgets
        run_script: Path to the script to execute when run button is clicked
        
    Returns:
        review_output: Output widget that displays the review summary
        run_button: Button widget to execute the script
        review_container: Container widget with all review elements
    """
    # Parameter port information for denormalization (same as in create_parameter_widgets)
    param_ports = {
        # PEQ - Low Shelf
        'low_shelf_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'low_shelf_cutoff': {'min': 20.0, 'max': 200.0, 'units': 'Hz'},
        'low_shelf_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - Band 1
        'band1_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'band1_cutoff': {'min': 200.0, 'max': 2000.0, 'units': 'Hz'},
        'band1_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - Band 2
        'band2_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'band2_cutoff': {'min': 200.0, 'max': 4000.0, 'units': 'Hz'},
        'band2_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - Band 3
        'band3_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'band3_cutoff': {'min': 2000.0, 'max': 8000.0, 'units': 'Hz'},
        'band3_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - Band 4
        'band4_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'band4_cutoff': {'min': 4000.0, 'max': 10800.0, 'units': 'Hz'},
        'band4_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # PEQ - High Shelf
        'high_shelf_gain': {'min': -24.0, 'max': 24.0, 'units': 'dB'},
        'high_shelf_cutoff': {'min': 4000.0, 'max': 10800.0, 'units': 'Hz'},
        'high_shelf_q': {'min': 0.1, 'max': 10.0, 'units': ''},
        # Compressor
        'threshold': {'min': -80.0, 'max': 0.0, 'units': 'dB'},
        'ratio': {'min': 1.0, 'max': 20.0, 'units': ''},
        'attack': {'min': 0.0001, 'max': 0.1, 'units': 's'},
        'release': {'min': 0.005, 'max': 1.0, 'units': 's'},
        'knee': {'min': 0.0, 'max': 12.0, 'units': 'dB'},
        'makeup_gain': {'min': -48.0, 'max': 48.0, 'units': 'dB'},
    }
    
    def denormalize_param(param_name, norm_value):
        """Denormalize a single parameter from [0,1] to actual range"""
        if param_name in param_ports:
            port = param_ports[param_name]
            denorm_value = denormalize(norm_value, port['max'], port['min'])
            return denorm_value, port['units']
        return norm_value, ''
    
    review_output = widgets.Output(layout=widgets.Layout(margin='10px 0px'))
    
    def update_review_display():
        """Update the review display with current inputs"""
        with review_output:
            review_output.clear_output(wait=True)
            mode = mode_dropdown.value
            
            print("=" * 60)
            print("📋 REVIEW YOUR SETTINGS")
            print("=" * 60)
            
            # Mode
            print(f"\n🎯 Selected Mode: {mode}")
            
            # Audio files
            print(f"\n🎵 Audio Files:")
            x_input = audio_widgets['X_input'].value
            if x_input:
                print(f"  ✓ X_input: {x_input}")
            else:
                print(f"  ✗ X_input: (not selected)")
            
            if mode in ['mode2', 'mode3']:
                y_input = audio_widgets['Y_input'].value
                if y_input:
                    print(f"  ✓ Y_input: {y_input}")
                else:
                    print(f"  ✗ Y_input: (not selected)")
            
            # Parameters (only for mode1 and mode2)
            if mode in ['mode1', 'mode2']:
                print(f"\n📊 DSP Parameters (normalized [0,1] → actual value):")
                
                # PEQ - Low Shelf
                gain_val = param_widgets['low_shelf_gain'].value
                gain_denorm, gain_units = denormalize_param('low_shelf_gain', gain_val)
                cutoff_val = param_widgets['low_shelf_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('low_shelf_cutoff', cutoff_val)
                q_val = param_widgets['low_shelf_q'].value
                q_denorm, q_units = denormalize_param('low_shelf_q', q_val)
                print(f"  PEQ - Low Shelf:")
                print(f"    Gain: {gain_val:.4f} → {gain_denorm:.2f} {gain_units}")
                print(f"    Cutoff: {cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}")
                print(f"    Q: {q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # PEQ - Band 1
                gain_val = param_widgets['band1_gain'].value
                gain_denorm, gain_units = denormalize_param('band1_gain', gain_val)
                cutoff_val = param_widgets['band1_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('band1_cutoff', cutoff_val)
                q_val = param_widgets['band1_q'].value
                q_denorm, q_units = denormalize_param('band1_q', q_val)
                print(f"  PEQ - Band 1:")
                print(f"    Gain: {gain_val:.4f} → {gain_denorm:.2f} {gain_units}")
                print(f"    Cutoff: {cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}")
                print(f"    Q: {q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # PEQ - Band 2
                gain_val = param_widgets['band2_gain'].value
                gain_denorm, gain_units = denormalize_param('band2_gain', gain_val)
                cutoff_val = param_widgets['band2_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('band2_cutoff', cutoff_val)
                q_val = param_widgets['band2_q'].value
                q_denorm, q_units = denormalize_param('band2_q', q_val)
                print(f"  PEQ - Band 2:")
                print(f"    Gain: {gain_val:.4f} → {gain_denorm:.2f} {gain_units}")
                print(f"    Cutoff: {cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}")
                print(f"    Q: {q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # PEQ - Band 3
                gain_val = param_widgets['band3_gain'].value
                gain_denorm, gain_units = denormalize_param('band3_gain', gain_val)
                cutoff_val = param_widgets['band3_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('band3_cutoff', cutoff_val)
                q_val = param_widgets['band3_q'].value
                q_denorm, q_units = denormalize_param('band3_q', q_val)
                print(f"  PEQ - Band 3:")
                print(f"    Gain: {gain_val:.4f} → {gain_denorm:.2f} {gain_units}")
                print(f"    Cutoff: {cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}")
                print(f"    Q: {q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # PEQ - Band 4
                gain_val = param_widgets['band4_gain'].value
                gain_denorm, gain_units = denormalize_param('band4_gain', gain_val)
                cutoff_val = param_widgets['band4_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('band4_cutoff', cutoff_val)
                q_val = param_widgets['band4_q'].value
                q_denorm, q_units = denormalize_param('band4_q', q_val)
                print(f"  PEQ - Band 4:")
                print(f"    Gain: {gain_val:.4f} → {gain_denorm:.2f} {gain_units}")
                print(f"    Cutoff: {cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}")
                print(f"    Q: {q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # PEQ - High Shelf
                gain_val = param_widgets['high_shelf_gain'].value
                gain_denorm, gain_units = denormalize_param('high_shelf_gain', gain_val)
                cutoff_val = param_widgets['high_shelf_cutoff'].value
                cutoff_denorm, cutoff_units = denormalize_param('high_shelf_cutoff', cutoff_val)
                q_val = param_widgets['high_shelf_q'].value
                q_denorm, q_units = denormalize_param('high_shelf_q', q_val)
                print(f"  PEQ - High Shelf:")
                print(f"    Gain: {gain_val:.4f} → {gain_denorm:.2f} {gain_units}")
                print(f"    Cutoff: {cutoff_val:.4f} → {cutoff_denorm:.1f} {cutoff_units}")
                print(f"    Q: {q_val:.4f} → {q_denorm:.3f} {q_units}")
                
                # Compressor
                thresh_val = param_widgets['threshold'].value
                thresh_denorm, thresh_units = denormalize_param('threshold', thresh_val)
                ratio_val = param_widgets['ratio'].value
                ratio_denorm, ratio_units = denormalize_param('ratio', ratio_val)
                attack_val = param_widgets['attack'].value
                attack_denorm, attack_units = denormalize_param('attack', attack_val)
                release_val = param_widgets['release'].value
                release_denorm, release_units = denormalize_param('release', release_val)
                knee_val = param_widgets['knee'].value
                knee_denorm, knee_units = denormalize_param('knee', knee_val)
                makeup_val = param_widgets['makeup_gain'].value
                makeup_denorm, makeup_units = denormalize_param('makeup_gain', makeup_val)
                print(f"  Compressor:")
                print(f"    Threshold: {thresh_val:.4f} → {thresh_denorm:.2f} {thresh_units}")
                print(f"    Ratio: {ratio_val:.4f} → {ratio_denorm:.2f} {ratio_units}")
                print(f"    Attack: {attack_val:.4f} → {attack_denorm:.4f} {attack_units}")
                print(f"    Release: {release_val:.4f} → {release_denorm:.3f} {release_units}")
                print(f"    Knee: {knee_val:.4f} → {knee_denorm:.2f} {knee_units}")
                print(f"    Makeup Gain: {makeup_val:.4f} → {makeup_denorm:.2f} {makeup_units}")
            else:
                print(f"\n📊 DSP Parameters: (Not used in Mode 3 - auto-inferred)")
            
            print("\n" + "=" * 60)
            print("Please review your settings above.")
            print("If everything looks correct, click the 'Run Experiment' button below.")
            print("=" * 60)
    
    # Update review when inputs change
    def on_input_change(change):
        update_review_display()
    
    mode_dropdown.observe(on_input_change, names='value')
    audio_widgets['X_input'].observe(on_input_change, names='value')
    if 'Y_input' in audio_widgets:
        audio_widgets['Y_input'].observe(on_input_change, names='value')
    for widget in param_widgets.values():
        widget.observe(on_input_change, names='value')
    
    # Initial display
    update_review_display()
    
    # Create run button
    run_output = widgets.Output(layout=widgets.Layout(margin='10px 0px'))
    
    def on_run_clicked(button):
        """Handle run button click - execute run.sh"""
        with run_output:
            run_output.clear_output(wait=True)
            print("🚀 Starting experiment...")
            
            # Validate inputs
            mode = mode_dropdown.value
            x_input = audio_widgets['X_input'].value
            
            if not x_input:
                print("❌ Error: X_input is not selected!")
                return
            
            if mode in ['mode2', 'mode3']:
                y_input = audio_widgets['Y_input'].value
                if not y_input:
                    print("❌ Error: Y_input is not selected!")
                    return
            else:
                y_input = None
            
            # Get parameters if needed
            params_dict = None
            if mode in ['mode1', 'mode2']:
                params_dict = {name: widget.value for name, widget in param_widgets.items()}
            
            # Determine output directory: mode_{i}/run_{n}
            mode_num = mode.replace('mode', '')
            mode_dir = f"mode{mode_num}"
            
            # Find next run number
            if os.path.exists(mode_dir):
                existing_runs = [d for d in os.listdir(mode_dir) 
                               if os.path.isdir(os.path.join(mode_dir, d)) and d.startswith('run_')]
                if existing_runs:
                    run_nums = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
                    next_run_num = max(run_nums) + 1 if run_nums else 1
                else:
                    next_run_num = 1
            else:
                os.makedirs(mode_dir, exist_ok=True)
                next_run_num = 1
            
            output_dir = os.path.join(mode_dir, f"run_{next_run_num}")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"📁 Output directory: {output_dir}")
            
            # Prepare inputs for run.sh
            script_dir = os.path.dirname(os.path.abspath(run_script))
            x_input_full = os.path.abspath(os.path.join('audios', x_input)) if x_input else ''
            y_input_full = os.path.abspath(os.path.join('audios', y_input)) if y_input else ''
            output_dir_full = os.path.abspath(output_dir)
            params_json = json.dumps(params_dict) if params_dict else ''
            
            # Check if run.sh exists
            run_script_full = os.path.abspath(run_script)
            if not os.path.exists(run_script_full):
                print(f"❌ Error: {run_script} not found!")
                return
            
            try:
                # Prepare environment variables
                # Note: Checkpoints are handled internally by process.py (both speech and music)
                env = os.environ.copy()
                env['MODE'] = mode
                env['X_INPUT'] = x_input_full
                if y_input:
                    env['Y_INPUT'] = y_input_full
                if params_dict:
                    env['PARAMS_JSON'] = params_json
                env['OUTPUT_DIR'] = output_dir_full
                
                # Execute run.sh in the background
                print(f"📝 Executing {run_script}...")
                print(f"   Mode: {mode}")
                print(f"   X_input: {x_input}")
                if y_input:
                    print(f"   Y_input: {y_input}")
                print(f"   Output: {output_dir}")
                
                process = subprocess.Popen(
                    ['bash', run_script_full],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=script_dir
                )
                
                print(f"✅ Experiment started! Process ID: {process.pid}")
                print(f"💡 The script is running in the background.")
                print(f"📁 Results will be saved to: {output_dir}")
                
            except Exception as e:
                print(f"❌ Error executing {run_script}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    run_button = widgets.Button(
        description='🚀 Run Experiment',
        button_style='success',
        tooltip='Click to execute run.sh with current settings',
        layout=widgets.Layout(width='200px', margin='10px 0px')
    )
    run_button.on_click(on_run_clicked)
    
    # Create review container
    review_container = widgets.VBox([
        widgets.HTML("<h3>📋 Review & Run</h3>"),
        review_output,
        widgets.HTML("<b>Ready to run?</b>"),
        widgets.HBox([run_button]),
        run_output
    ])
    
    # Display the container
    display(review_container)
    
    return review_output, run_button, review_container, run_output

