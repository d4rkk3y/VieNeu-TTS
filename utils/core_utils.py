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

def split_text_into_chunks(
    text: str, 
    max_chars: int = 256, 
    min_chars: int = 50
):
    """
    Chunks text for TTS by respecting natural pause boundaries (.,!?;:—).
    Adds periods only if a hard word-cut is required.
    """
    
    # --- Step 1: Split text into "Breath Groups" / Sentences ---
    # We split by: . ! ? … (Sentence terminators)
    # And also: , ; : — (Phrase/Pause markers)
    # This creates smaller atoms, allowing the merger to fill max_chars more efficiently.
    
    # Regex explanation:
    # (?<=...) : Lookbehind (checks if preceded by)
    # [.!?…,;:—] : Matches any of these punctuation marks
    # \s+ : Matches one or more whitespace characters
    raw_sentences = re.split(r'(?<=[.!?…,;:—])\s+', text)
    
    # Clean empty strings
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    if not sentences:
        return {"chunks": [], "flags": []}

    # --- Step 2: Create Atoms ---
    # An atom is a text block + metadata.
    # Metadata: 
    #   - is_continuation: Is this atom a continuation of the previous atom's sentence?
    #   - requires_period: Does this atom need a period IF it ends a chunk (hard cut)?
    
    atoms = []

    for sent in sentences:
        # Check if the "breath group" itself fits in max_chars
        if len(sent) <= max_chars:
            atoms.append({
                'text': sent,
                'is_continuation': False, # Natural break, so treated as new/clean start context
                'requires_period': False  # Already has punctuation (.,;:)
            })
        else:
            # Hard split needed (Breath group is still too huge)
            words = sent.split(' ')
            current_buffer = ""
            first_split = True
            
            for word in words:
                # Reserve 1 char for potential period
                space_needed = 1 if current_buffer else 0
                
                if len(current_buffer) + space_needed + len(word) + 1 <= max_chars:
                    current_buffer += (" " if current_buffer else "") + word
                else:
                    # Buffer full. Create atom.
                    atoms.append({
                        'text': current_buffer,
                        'is_continuation': not first_split,
                        'requires_period': True # Hard cut -> Needs period
                    })
                    first_split = False
                    current_buffer = word
            
            # Append the remainder
            if current_buffer:
                atoms.append({
                    'text': current_buffer,
                    'is_continuation': True, 
                    'requires_period': False # Ends with original punctuation
                })

    # --- Step 3: Merge Atoms into Chunks ---
    
    final_chunks = []
    final_flags = []
    
    current_chunk_atoms = [] 
    current_chunk_len = 0    
    
    for atom in atoms:
        text_len = len(atom['text'])
        prefix_len = 1 if current_chunk_atoms else 0
        
        # Cost check: Current + Space + NewAtom + (Period if NewAtom needs it and ends chunk)
        # Note: We don't know if NewAtom ends the chunk yet, but we must reserve space just in case.
        potential_period_len = 1 if atom['requires_period'] else 0
        
        total_cost = current_chunk_len + prefix_len + text_len + potential_period_len
        
        if total_cost <= max_chars:
            # Fits in current chunk
            current_chunk_atoms.append(atom)
            current_chunk_len += prefix_len + text_len
        else:
            # Does not fit. Finalize current chunk.
            if current_chunk_atoms:
                # Construct text
                chunk_str = ""
                for i, a in enumerate(current_chunk_atoms):
                    prefix = " " if i > 0 else ""
                    chunk_str += prefix + a['text']
                
                # Check last atom for period requirement
                if current_chunk_atoms[-1]['requires_period']:
                    chunk_str += "."
                
                # Flag logic: Based on the start of the chunk
                first_atom_is_cont = current_chunk_atoms[0]['is_continuation']
                
                final_chunks.append(chunk_str)
                final_flags.append(1 if first_atom_is_cont else 0)
            
            # Start new chunk with current atom
            current_chunk_atoms = [atom]
            current_chunk_len = len(atom['text'])

    # --- Step 4: Handle Remainder ---
    if current_chunk_atoms:
        chunk_str = ""
        for i, a in enumerate(current_chunk_atoms):
            prefix = " " if i > 0 else ""
            chunk_str += prefix + a['text']
        
        if current_chunk_atoms[-1]['requires_period']:
            chunk_str += "."
            
        first_atom_is_cont = current_chunk_atoms[0]['is_continuation']
        final_chunks.append(chunk_str)
        final_flags.append(1 if first_atom_is_cont else 0)
        
    return {
        "chunks": final_chunks,
        "flags": final_flags
    }

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")