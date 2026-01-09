import re
import os
from typing import List, Tuple
import librosa
import numpy as np

def process_audio_sequence(
    audio_list: list[np.ndarray], 
    flags: list[int], 
    sr: int, 
    crossfade_ms: int = 30, 
    silence_thresh_db: int = 40,
    silence_pad_ms: int = 150
) -> np.ndarray:
    """
    Concatenates a list of audio clips (numpy arrays) based on flags.
    
    Args:
        audio_list (List[np.ndarray]): List of 1D numpy arrays (audio clips).
        flags (List[int]): 0 = New sentence (insert silence), 1 = Continuation (cross-fade).
        sr (int): Sample rate.
        crossfade_ms (int): Overlap duration for flag=1 joins.
        silence_thresh_db (int): Silence removal threshold for flag=1 joins.
        silence_pad_ms (int): Duration of silence (in ms) to insert when flag=0.
                              (Adds a pause between sentences).
        
    Returns:
        np.ndarray: Single concatenated 1D audio array.
    """
    
    if not audio_list:
        return np.array([])
        
    # Initialize the output list with the first clip
    # We use a list to store chunks and concat ONCE at the end for performance.
    output_chunks = [audio_list[0]]
    
    # Pre-calculate durations in samples
    n_crossfade = int(sr * (crossfade_ms / 1000.0))
    n_pad_silence = int(sr * (silence_pad_ms / 1000.0))

    for i in range(1, len(audio_list)):
        current_clip = audio_list[i]
        method = flags[i]
        
        if method == 0:
            # === NEW SENTENCE (Insert Silence) ===
            # Flag 0 means the previous clip finished a sentence and this is a new one.
            # We add a pause to make it sound natural.
            
            if n_pad_silence > 0:
                # Create silence array matching the dtype of the audio (usually float32)
                silence_arr = np.zeros(n_pad_silence, dtype=current_clip.dtype)
                output_chunks.append(silence_arr)
            
            # Append the new clip directly
            output_chunks.append(current_clip)
            
        elif method == 1:
            # === BROKEN SENTENCE (Trim + Cross-fade) ===
            # Flag 1 means this clip continues the previous one (split by word).
            # We must remove silence and overlap them to hide the cut.
            
            # 1. Get the previous chunk
            prev_clip = output_chunks[-1]
            
            # 2. Trim Silence
            # Trim TRAILING silence of previous clip
            _, prev_idx = librosa.effects.trim(prev_clip, top_db=silence_thresh_db)
            prev_trimmed = prev_clip[:prev_idx[1]]
            
            # Trim LEADING silence of current clip
            _, curr_idx = librosa.effects.trim(current_clip, top_db=silence_thresh_db)
            curr_trimmed = current_clip[curr_idx[0]:]
            
            # 3. Check length safety
            if len(prev_trimmed) < n_crossfade or len(curr_trimmed) < n_crossfade:
                # If too short to fade, fallback to just appending trimmed versions
                output_chunks[-1] = prev_trimmed
                output_chunks.append(curr_trimmed)
                continue

            # 4. Generate Cross-fade parts
            
            # Split previous into Body and Fade-Out
            prev_body = prev_trimmed[:-n_crossfade]
            fade_out = prev_trimmed[-n_crossfade:]
            
            # Split current into Fade-In and Body
            fade_in = curr_trimmed[:n_crossfade]
            curr_body = curr_trimmed[n_crossfade:]
            
            # Create the crossfade bridge (Linear interpolation)
            fade_curve = np.linspace(0, 1, n_crossfade)
            crossfade_bridge = (fade_out * (1 - fade_curve)) + (fade_in * fade_curve)
            
            # 5. Update the Output List
            # Replace the last element (previous clip) with its shortened body
            output_chunks[-1] = prev_body
            # Add the blended bridge
            output_chunks.append(crossfade_bridge)
            # Add the new clip body
            output_chunks.append(curr_body)

    # Final merge of all chunks
    return np.concatenate(output_chunks)

def split_text_into_chunks(text: str, max_chars: int = 256) -> Tuple[List[str], List[int]]:
    """
    Split raw text into chunks no longer than max_chars.
    
    If a sentence is split by words (because it exceeds max_chars), 
    a period '.' is added to the end of the split chunk to help TTS processing,
    even though the flag is set to 1 (continuation).

    Returns:
        (chunks, flags): 
        - chunks: List of text strings.
        - flags: List of ints. 
                 1: Chunk is a continuation (split by word).
                 0: Chunk is a new sentence start.
    """
    # Split by sentence endings (. ! ? ...) or newlines
    sentences = re.split(r"(?<=[\.\!\?\…\n])\s+|(?<=\n)", text.strip())
    
    chunks: List[str] = []
    flags: List[int] = []
    
    buffer = ""
    next_chunk_is_continuation = 0

    def flush_buffer():
        nonlocal buffer, next_chunk_is_continuation
        if buffer:
            chunks.append(buffer.strip())
            flags.append(next_chunk_is_continuation)
            buffer = ""
            next_chunk_is_continuation = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # --- CASE 1: Sentence fits within max_chars ---
        if len(sentence) <= max_chars:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                flush_buffer()
                buffer = sentence
                next_chunk_is_continuation = 0
            continue

        # --- CASE 2: Sentence is HUGE (larger than max_chars) ---
        flush_buffer()
        
        words = sentence.split()
        current_segment = ""
        current_flag = 0 

        for word in words:
            candidate = f"{current_segment} {word}".strip() if current_segment else word
            
            if len(candidate) > max_chars and current_segment:
                # 1. Add the current full segment WITH A PERIOD
                # We intentionally add '.' here per requirement
                chunk_to_add = current_segment.strip() + "." 
                
                chunks.append(chunk_to_add)
                flags.append(current_flag)
                
                # 2. Prepare for the next segment
                current_segment = word
                current_flag = 1 
            else:
                current_segment = candidate
        
        # Add the final piece of the long sentence
        if current_segment:
            chunks.append(current_segment.strip())
            flags.append(current_flag)
            
        next_chunk_is_continuation = 0

    flush_buffer()
    
    return chunks, flags

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")