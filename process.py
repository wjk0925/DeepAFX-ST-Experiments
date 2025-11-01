#!/usr/bin/env python
"""
Process audio according to user inputs from the interactive notebook.
Handles three modes:
- Mode 1: Apply P_groundtruth to X_input → X_processed, then run DeepAFX-ST
- Mode 2: Apply P to Y_input → Y_processed, then run DeepAFX-ST
- Mode 3: Run DeepAFX-ST with X_input and Y_input (no parameters)
"""

import os
import sys
import json
import torch
import resampy
import argparse
import torchaudio
import numpy as np

from deepafx_st.utils import DSPMode
from deepafx_st.system import System
from deepafx_st.processors.autodiff.channel import AutodiffChannel


def load_and_prepare_audio(audio_path, sample_rate=24000, max_length_sec=5, target_db=-12.0):
    """
    Load audio, resample to target sample rate, normalize to target dB.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 24000)
        max_length_sec: Maximum length in seconds to process (default 5)
        target_db: Target peak level in dB (default -12.0)
        
    Returns:
        audio: Processed audio tensor [1, 1, samples]
        original_sr: Original sample rate
    """
    # Load audio
    audio, original_sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if original_sr != sample_rate:
        print(f"Resampling {audio_path} from {original_sr} Hz to {sample_rate} Hz...")
        audio_24000 = torch.tensor(resampy.resample(audio.view(-1).numpy(), original_sr, sample_rate))
        audio_24000 = audio_24000.view(1, -1)
    else:
        audio_24000 = audio
    
    # Limit to max_length_sec
    max_samples = sample_rate * max_length_sec
    if audio_24000.shape[1] > max_samples:
        audio_24000 = audio_24000[:, :max_samples]
    
    # Peak normalize to target dB
    if audio_24000.abs().max() > 0:
        audio_24000 /= audio_24000.abs().max()
        audio_24000 *= 10 ** (target_db / 20.0)
    
    # Reshape to [1, 1, samples]
    audio_24000 = audio_24000.view(1, 1, -1)
    
    return audio_24000, original_sr


def apply_dsp_parameters(audio, params_normalized, sample_rate=24000):
    """
    Apply DSP parameters (PEQ + Compressor) to audio.
    
    Args:
        audio: Audio tensor [1, 1, samples]
        params_normalized: Dictionary of normalized parameters [0,1]
        sample_rate: Sample rate (default 24000)
        
    Returns:
        processed_audio: Processed audio tensor
    """
    # Create AutodiffChannel processor
    processor = AutodiffChannel(sample_rate)
    
    # Convert normalized parameters to tensor in the correct order
    # PEQ parameters (18): low_shelf (3), band1-4 (12), high_shelf (3)
    peq_params = [
        params_normalized['low_shelf_gain'],
        params_normalized['low_shelf_cutoff'],
        params_normalized['low_shelf_q'],
        params_normalized['band1_gain'],
        params_normalized['band1_cutoff'],
        params_normalized['band1_q'],
        params_normalized['band2_gain'],
        params_normalized['band2_cutoff'],
        params_normalized['band2_q'],
        params_normalized['band3_gain'],
        params_normalized['band3_cutoff'],
        params_normalized['band3_q'],
        params_normalized['band4_gain'],
        params_normalized['band4_cutoff'],
        params_normalized['band4_q'],
        params_normalized['high_shelf_gain'],
        params_normalized['high_shelf_cutoff'],
        params_normalized['high_shelf_q'],
    ]
    
    # Compressor parameters (6)
    comp_params = [
        params_normalized['threshold'],
        params_normalized['ratio'],
        params_normalized['attack'],
        params_normalized['release'],
        params_normalized['knee'],
        params_normalized['makeup_gain'],
    ]
    
    # Combine all parameters
    all_params = peq_params + comp_params
    params_tensor = torch.tensor(all_params, dtype=torch.float32).view(1, -1)
    
    # Apply processing
    with torch.no_grad():
        processed_audio = processor(audio, params_tensor, sample_rate=sample_rate)
    
    return processed_audio


def run_deepafx_st(x_audio, r_audio, checkpoint_path, gpu=False):
    """
    Run DeepAFX-ST model inference.
    
    Args:
        x_audio: Input audio tensor [1, 1, samples]
        r_audio: Reference audio tensor [1, 1, samples]
        checkpoint_path: Path to model checkpoint
        gpu: Use GPU if available
        
    Returns:
        y_hat: Predicted output audio
        p_predicted: Predicted parameters
        e_x: Input embedding
    """
    # Load model
    if "proxy" in checkpoint_path:
        logdir = os.path.dirname(os.path.dirname(checkpoint_path))
        pckpts = 'checkpoints'
        if 'proxy0m' in logdir or 'proxy2m' in logdir:
            peq_ckpt = os.path.join(pckpts, "proxies/jamendo/peq/lightning_logs/version_0/checkpoints/epoch=326-step=204374-val-jamendo-peq.ckpt")
            comp_ckpt = os.path.join(pckpts, "proxies/jamendo/comp/lightning_logs/version_0/checkpoints/epoch=274-step=171874-val-jamendo-comp.ckpt")
        else:
            peq_ckpt = os.path.join(pckpts, "proxies/libritts/peq/lightning_logs/version_1/checkpoints/epoch=111-step=139999-val-libritts-peq.ckpt")
            comp_ckpt = os.path.join(pckpts, "proxies/libritts/comp/lightning_logs/version_1/checkpoints/epoch=255-step=319999-val-libritts-comp.ckpt")
        
        proxy_ckpts = [peq_ckpt, comp_ckpt]
        system = System.load_from_checkpoint(
            checkpoint_path, dsp_mode=DSPMode.INFER, proxy_ckpts=proxy_ckpts
        ).eval()
    else:
        system = System.load_from_checkpoint(
            checkpoint_path, dsp_mode=DSPMode.NONE, batch_size=1
        ).eval()
    
    if gpu and torch.cuda.is_available():
        system = system.to("cuda")
        x_audio = x_audio.to("cuda")
        r_audio = r_audio.to("cuda")
    
    # Run inference
    with torch.no_grad():
        y_hat, p_predicted, e_x = system(x_audio, r_audio)
    
    # Cleanup
    if hasattr(system, 'shutdown'):
        system.shutdown()
    
    return y_hat, p_predicted, e_x


def main():
    parser = argparse.ArgumentParser(description='Process audio with DeepAFX-ST based on user inputs')
    
    # Mode and audio files
    parser.add_argument('--mode', type=str, required=True, choices=['mode1', 'mode2', 'mode3'],
                        help='Processing mode')
    parser.add_argument('--x_input', type=str, required=True,
                        help='Path to X_input audio file')
    parser.add_argument('--y_input', type=str, default=None,
                        help='Path to Y_input audio file (required for mode2 and mode3)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (mode_{i}/run_{n})')
    
    # Parameters (for mode1 and mode2)
    parser.add_argument('--params', type=str, default=None,
                        help='JSON string with normalized parameters [0,1]')
    
    # Model checkpoints (default to both speech and music autodiff models)
    parser.add_argument('--checkpoint_speech', type=str, default=None,
                        help='Path to DeepAFX-ST speech checkpoint')
    parser.add_argument('--checkpoint_music', type=str, default=None,
                        help='Path to DeepAFX-ST music checkpoint')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Parse parameters if provided
    params_normalized = None
    if args.params:
        params_normalized = json.loads(args.params)
    
    # Validate inputs based on mode
    if args.mode in ['mode2', 'mode3']:
        if not args.y_input:
            raise ValueError(f"Y_input is required for {args.mode}")
    
    if args.mode in ['mode1', 'mode2']:
        if not params_normalized:
            raise ValueError(f"Parameters are required for {args.mode}")
    
    # Set default checkpoint paths if not provided
    if args.checkpoint_speech is None:
        args.checkpoint_speech = "checkpoints/style/libritts/autodiff/lightning_logs/version_1/checkpoints/epoch=367-step=1226911-val-libritts-autodiff.ckpt"
    if args.checkpoint_music is None:
        args.checkpoint_music = "checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-autodiff.ckpt"
    
    # Validate checkpoint paths exist
    checkpoints_info = {}
    if os.path.exists(args.checkpoint_speech):
        checkpoints_info['speech'] = os.path.abspath(args.checkpoint_speech)
    else:
        print(f"Warning: Speech checkpoint not found: {args.checkpoint_speech}")
    
    if os.path.exists(args.checkpoint_music):
        checkpoints_info['music'] = os.path.abspath(args.checkpoint_music)
    else:
        print(f"Warning: Music checkpoint not found: {args.checkpoint_music}")
    
    if not checkpoints_info:
        raise ValueError("At least one checkpoint must be available!")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare X_input
    print(f"Loading X_input: {args.x_input}")
    x_audio, x_original_sr = load_and_prepare_audio(args.x_input)
    
    # Store original file paths
    inputs_info = {
        'mode': args.mode,
        'x_input_original': os.path.abspath(args.x_input),
        'x_input_filename': os.path.basename(args.x_input),
        'x_input_original_sr': x_original_sr,
    }
    
    # Initialize normalization factors tracking
    normalization_factors = {}
    
    # Mode 1: Apply P_groundtruth to X_input → X_processed, then run DeepAFX-ST
    if args.mode == 'mode1':
        print("Mode 1: Parameter Restoration")
        
        # Apply parameters to X_input to create X_processed
        print("Applying ground truth parameters to X_input...")
        x_processed = apply_dsp_parameters(x_audio, params_normalized)
        
        # Save X_processed (both unnormalized and normalized versions)
        x_processed_flat = x_processed.view(1, -1)
        x_processed_max = x_processed_flat.abs().max().item()
        normalization_factors['x_processed'] = x_processed_max
        
        # Save unnormalized version
        x_processed_path = os.path.join(args.output_dir, 'x_processed.wav')
        torchaudio.save(x_processed_path, x_processed_flat.cpu(), 24000)
        inputs_info['x_processed_file'] = 'x_processed.wav'
        
        # Save normalized version
        x_processed_save = x_processed_flat / x_processed_max
        x_processed_normalized_path = os.path.join(args.output_dir, 'x_processed_normalized.wav')
        torchaudio.save(x_processed_normalized_path, x_processed_save.cpu(), 24000)
        inputs_info['x_processed_normalized_file'] = 'x_processed_normalized.wav'
        
        # Store reference info
        inputs_info['reference_type'] = 'x_processed'
        inputs_info['reference_file'] = 'x_processed.wav'
        
        # Will run both models after processing - store x_processed for later
        reference_audio = x_processed
    
    # Mode 2: Apply P to Y_input → Y_processed, then run DeepAFX-ST
    elif args.mode == 'mode2':
        print("Mode 2: Cross-Audio Style Transfer with Known Parameters")
        
        # Load Y_input
        print(f"Loading Y_input: {args.y_input}")
        y_audio, y_original_sr = load_and_prepare_audio(args.y_input)
        inputs_info['y_input_original'] = os.path.abspath(args.y_input)
        inputs_info['y_input_filename'] = os.path.basename(args.y_input)
        inputs_info['y_input_original_sr'] = y_original_sr
        
        # Apply parameters to Y_input to create Y_processed
        print("Applying parameters to Y_input...")
        y_processed = apply_dsp_parameters(y_audio, params_normalized)
        
        # Save Y_processed (both unnormalized and normalized versions)
        y_processed_flat = y_processed.view(1, -1)
        y_processed_max = y_processed_flat.abs().max().item()
        normalization_factors['y_processed'] = y_processed_max
        
        # Save unnormalized version
        y_processed_path = os.path.join(args.output_dir, 'y_processed.wav')
        torchaudio.save(y_processed_path, y_processed_flat.cpu(), 24000)
        inputs_info['y_processed_file'] = 'y_processed.wav'
        
        # Save normalized version
        y_processed_save = y_processed_flat / y_processed_max
        y_processed_normalized_path = os.path.join(args.output_dir, 'y_processed_normalized.wav')
        torchaudio.save(y_processed_normalized_path, y_processed_save.cpu(), 24000)
        inputs_info['y_processed_normalized_file'] = 'y_processed_normalized.wav'
        
        # Store reference info
        inputs_info['reference_type'] = 'y_processed'
        inputs_info['reference_file'] = 'y_processed.wav'
        
        # Will run both models after processing - store y_processed for later
        reference_audio = y_processed
    
    # Mode 3: Run DeepAFX-ST with X_input and Y_input directly
    else:  # mode3
        print("Mode 3: Automatic Style Transfer")
        
        # Load Y_input
        print(f"Loading Y_input: {args.y_input}")
        y_audio, y_original_sr = load_and_prepare_audio(args.y_input)
        inputs_info['y_input_original'] = os.path.abspath(args.y_input)
        inputs_info['y_input_filename'] = os.path.basename(args.y_input)
        inputs_info['y_input_original_sr'] = y_original_sr
        
        # Store reference info
        inputs_info['reference_type'] = 'y_input'
        inputs_info['reference_file'] = 'y_input.wav'
        
        # Will run both models after processing - store y_audio for later
        reference_audio = y_audio
    
    # Prepare audio for saving (both unnormalized and normalized)
    x_audio_flat = x_audio.view(1, -1)
    x_audio_max = x_audio_flat.abs().max().item()
    
    # Save outputs
    print(f"Saving outputs to {args.output_dir}...")
    
    # Save X_input (both unnormalized and normalized versions)
    # Unnormalized version (still has -12dB peak normalization from load_and_prepare_audio)
    x_input_path = os.path.join(args.output_dir, 'x_input.wav')
    torchaudio.save(x_input_path, x_audio_flat.cpu(), 24000)
    inputs_info['x_input_file'] = 'x_input.wav'
    
    # Normalized version
    x_audio_save = x_audio_flat / x_audio_max
    x_input_normalized_path = os.path.join(args.output_dir, 'x_input_normalized.wav')
    torchaudio.save(x_input_normalized_path, x_audio_save.cpu(), 24000)
    inputs_info['x_input_normalized_file'] = 'x_input_normalized.wav'
    
    # Save Y_input if used (both unnormalized and normalized versions)
    if args.mode in ['mode2', 'mode3']:
        y_audio_flat = y_audio.view(1, -1)
        y_audio_max = y_audio_flat.abs().max().item()
        
        # Unnormalized version (still has -12dB peak normalization from load_and_prepare_audio)
        y_input_path = os.path.join(args.output_dir, 'y_input.wav')
        torchaudio.save(y_input_path, y_audio_flat.cpu(), 24000)
        inputs_info['y_input_file'] = 'y_input.wav'
        
        # Normalized version
        y_audio_save = y_audio_flat / y_audio_max
        y_input_normalized_path = os.path.join(args.output_dir, 'y_input_normalized.wav')
        torchaudio.save(y_input_normalized_path, y_audio_save.cpu(), 24000)
        inputs_info['y_input_normalized_file'] = 'y_input_normalized.wav'
    
    # Run both speech and music models
    print("\n" + "="*60)
    print("Running DeepAFX-ST models...")
    print("="*60)
    
    predictions = {}
    predictions_info = {}
    
    # Import processors for denormalizing predicted parameters
    from deepafx_st.processors.autodiff.peq import ParametricEQ
    from deepafx_st.processors.autodiff.compressor import Compressor
    peq = ParametricEQ(24000)
    comp = Compressor(24000)
    peq_num_params = peq.num_control_params
    
    # Run speech model if available
    if 'speech' in checkpoints_info:
        print(f"\n[1/2] Running Speech Model...")
        print(f"Checkpoint: {checkpoints_info['speech']}")
        try:
            y_hat_speech, p_predicted_speech, e_x_speech = run_deepafx_st(
                x_audio, reference_audio, checkpoints_info['speech'], args.gpu
            )
            
            # Record normalization factor and save (both unnormalized and normalized versions)
            y_hat_speech_flat = y_hat_speech.view(1, -1)
            y_hat_speech_max = y_hat_speech_flat.abs().max().item()
            normalization_factors['x_predicted_speech'] = y_hat_speech_max
            
            # Save unnormalized version
            y_predicted_speech_path = os.path.join(args.output_dir, 'x_predicted_speech.wav')
            torchaudio.save(y_predicted_speech_path, y_hat_speech_flat.cpu(), 24000)
            
            # Save normalized version
            y_hat_speech_save = y_hat_speech_flat / y_hat_speech_max
            y_predicted_speech_normalized_path = os.path.join(args.output_dir, 'x_predicted_speech_normalized.wav')
            torchaudio.save(y_predicted_speech_normalized_path, y_hat_speech_save.cpu(), 24000)
            
            # Denormalize predicted parameters
            p_predicted_speech_denorm = None
            if p_predicted_speech is not None:
                p_pred_speech_np = p_predicted_speech.cpu().numpy().flatten()
                p_peq_speech = torch.tensor(p_pred_speech_np[:peq_num_params])
                p_comp_speech = torch.tensor(p_pred_speech_np[peq_num_params:])
                
                peq_params_denorm_speech = peq.denormalize_params(p_peq_speech)
                comp_params_denorm_speech = comp.denormalize_params(p_comp_speech)
                
                # Convert to dictionary
                p_predicted_speech_denorm = {}
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
                    p_predicted_speech_denorm[name] = float(peq_params_denorm_speech[i])
                
                for i, name in enumerate(comp_param_names):
                    p_predicted_speech_denorm[name] = float(comp_params_denorm_speech[i])
            
            predictions['speech'] = y_hat_speech
            predictions_info['speech'] = {
                'checkpoint_path': checkpoints_info['speech'],
                'output_file': 'x_predicted_speech.wav',
                'output_file_normalized': 'x_predicted_speech_normalized.wav',
                'p_predicted_normalized': p_predicted_speech.cpu().numpy().flatten().tolist() if p_predicted_speech is not None else None,
                'p_predicted_denormalized': p_predicted_speech_denorm
            }
            print(f"✓ Speech model output saved: x_predicted_speech.wav (unnormalized) and x_predicted_speech_normalized.wav")
        except Exception as e:
            print(f"✗ Error running speech model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run music model if available
    if 'music' in checkpoints_info:
        print(f"\n[2/2] Running Music Model...")
        print(f"Checkpoint: {checkpoints_info['music']}")
        try:
            y_hat_music, p_predicted_music, e_x_music = run_deepafx_st(
                x_audio, reference_audio, checkpoints_info['music'], args.gpu
            )
            
            # Record normalization factor and save (both unnormalized and normalized versions)
            y_hat_music_flat = y_hat_music.view(1, -1)
            y_hat_music_max = y_hat_music_flat.abs().max().item()
            normalization_factors['x_predicted_music'] = y_hat_music_max
            
            # Save unnormalized version
            y_predicted_music_path = os.path.join(args.output_dir, 'x_predicted_music.wav')
            torchaudio.save(y_predicted_music_path, y_hat_music_flat.cpu(), 24000)
            
            # Save normalized version
            y_hat_music_save = y_hat_music_flat / y_hat_music_max
            y_predicted_music_normalized_path = os.path.join(args.output_dir, 'x_predicted_music_normalized.wav')
            torchaudio.save(y_predicted_music_normalized_path, y_hat_music_save.cpu(), 24000)
            
            # Denormalize predicted parameters
            p_predicted_music_denorm = None
            if p_predicted_music is not None:
                p_pred_music_np = p_predicted_music.cpu().numpy().flatten()
                p_peq_music = torch.tensor(p_pred_music_np[:peq_num_params])
                p_comp_music = torch.tensor(p_pred_music_np[peq_num_params:])
                
                peq_params_denorm_music = peq.denormalize_params(p_peq_music)
                comp_params_denorm_music = comp.denormalize_params(p_comp_music)
                
                # Convert to dictionary
                p_predicted_music_denorm = {}
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
                    p_predicted_music_denorm[name] = float(peq_params_denorm_music[i])
                
                for i, name in enumerate(comp_param_names):
                    p_predicted_music_denorm[name] = float(comp_params_denorm_music[i])
            
            predictions['music'] = y_hat_music
            predictions_info['music'] = {
                'checkpoint_path': checkpoints_info['music'],
                'output_file': 'x_predicted_music.wav',
                'output_file_normalized': 'x_predicted_music_normalized.wav',
                'p_predicted_normalized': p_predicted_music.cpu().numpy().flatten().tolist() if p_predicted_music is not None else None,
                'p_predicted_denormalized': p_predicted_music_denorm
            }
            print(f"✓ Music model output saved: x_predicted_music.wav (unnormalized) and x_predicted_music_normalized.wav")
        except Exception as e:
            print(f"✗ Error running music model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not predictions:
        raise RuntimeError("No models were successfully run!")
    
    # Save parameters (for mode1 and mode2)
    if params_normalized:
        inputs_info['parameters_normalized'] = params_normalized
        # Also save as denormalized for reference (peq and comp already initialized above)
        
        # Get all parameters in order
        peq_params_norm = torch.tensor([
            params_normalized['low_shelf_gain'],
            params_normalized['low_shelf_cutoff'],
            params_normalized['low_shelf_q'],
            params_normalized['band1_gain'],
            params_normalized['band1_cutoff'],
            params_normalized['band1_q'],
            params_normalized['band2_gain'],
            params_normalized['band2_cutoff'],
            params_normalized['band2_q'],
            params_normalized['band3_gain'],
            params_normalized['band3_cutoff'],
            params_normalized['band3_q'],
            params_normalized['band4_gain'],
            params_normalized['band4_cutoff'],
            params_normalized['band4_q'],
            params_normalized['high_shelf_gain'],
            params_normalized['high_shelf_cutoff'],
            params_normalized['high_shelf_q'],
        ])
        
        comp_params_norm = torch.tensor([
            params_normalized['threshold'],
            params_normalized['ratio'],
            params_normalized['attack'],
            params_normalized['release'],
            params_normalized['knee'],
            params_normalized['makeup_gain'],
        ])
        
        peq_params_denorm = peq.denormalize_params(peq_params_norm)
        comp_params_denorm = comp.denormalize_params(comp_params_norm)
        
        # Convert to lists for JSON
        params_denorm_dict = {}
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
            params_denorm_dict[name] = float(peq_params_denorm[i])
        
        for i, name in enumerate(comp_param_names):
            params_denorm_dict[name] = float(comp_params_denorm[i])
        
        inputs_info['parameters_denormalized'] = params_denorm_dict
    
    # Store normalization factors (for restoring original audio levels)
    inputs_info['normalization_factors'] = normalization_factors
    # Note: normalization_factors contains the max absolute values used to normalize:
    # - x_processed (mode1): max value before normalization
    # - y_processed (mode2): max value before normalization  
    # - x_predicted_speech: max value before normalization
    # - x_predicted_music: max value before normalization
    # To restore: audio_restored = audio_normalized * normalization_factors[filename]
    print(f"\n📊 Normalization factors stored: {normalization_factors}")
    
    # Store model information and predictions
    inputs_info['models_used'] = predictions_info
    inputs_info['checkpoints'] = checkpoints_info
    
    # Save all inputs and metadata
    metadata_path = os.path.join(args.output_dir, 'inputs_and_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(inputs_info, f, indent=2)
    
    print(f"✓ Processing complete!")
    print(f"✓ Outputs saved to: {args.output_dir}")
    print(f"✓ Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()

