import re
import os
import librosa
import numpy as np

def process_audio_sequence(
    audio_list: list[np.ndarray], 
    flags: list[int], 
    sr: int, 
    crossfade_ms: int = 30, 
    silence_thresh_db: int = 40,
    silence_pad_ms: int = 150,
    comma_pad_ms: int = 50  # New parameter for comma/phrase pauses
) -> np.ndarray:
    """
    Concatenates a list of audio clips based on flags.
    
    Args:
        audio_list: List of audio arrays.
        flags: 
            0 = New Sentence (Long Silence: silence_pad_ms)
            1 = Continuation/Hard Cut (Cross-fade)
            2 = Phrase/Comma (Short Silence: comma_pad_ms)
        sr: Sample rate.
        crossfade_ms: Duration of overlap for Flag 1.
        silence_thresh_db: Threshold for trimming for Flag 1.
        silence_pad_ms: Duration of silence for Flag 0.
        comma_pad_ms: Duration of silence for Flag 2.
    """
    
    if not audio_list:
        return np.array([])
        
    output_chunks = [audio_list[0]]
    
    # Pre-calculate durations in samples
    n_crossfade = int(sr * (crossfade_ms / 1000.0))
    n_pad_sentence = int(sr * (silence_pad_ms / 1000.0))
    n_pad_comma = int(sr * (comma_pad_ms / 1000.0))

    for i in range(1, len(audio_list)):
        current_clip = audio_list[i]
        method = flags[i]
        
        if method == 0:
            # === NEW SENTENCE (Flag 0) ===
            # End of sentence (. ! ?). Insert long silence.
            if n_pad_sentence > 0:
                silence_arr = np.zeros(n_pad_sentence, dtype=current_clip.dtype)
                output_chunks.append(silence_arr)
            output_chunks.append(current_clip)

        elif method == 2:
            # === PHRASE/PAUSE (Flag 2) ===
            # End of phrase (, : -). Insert short silence.
            if n_pad_comma > 0:
                silence_arr = np.zeros(n_pad_comma, dtype=current_clip.dtype)
                output_chunks.append(silence_arr)
            output_chunks.append(current_clip)
            
        elif method == 1:
            # === BROKEN SENTENCE (Flag 1) ===
            # Hard cut due to length. Trim and Cross-fade.
            
            prev_clip = output_chunks[-1]
            
            # Trim silence
            _, prev_idx = librosa.effects.trim(prev_clip, top_db=silence_thresh_db)
            prev_trimmed = prev_clip[:prev_idx[1]]
            
            _, curr_idx = librosa.effects.trim(current_clip, top_db=silence_thresh_db)
            curr_trimmed = current_clip[curr_idx[0]:]
            
            # Safety check
            if len(prev_trimmed) < n_crossfade or len(curr_trimmed) < n_crossfade:
                output_chunks[-1] = prev_trimmed
                output_chunks.append(curr_trimmed)
                continue

            # Generate Cross-fade
            prev_body = prev_trimmed[:-n_crossfade]
            fade_out = prev_trimmed[-n_crossfade:]
            
            fade_in = curr_trimmed[:n_crossfade]
            curr_body = curr_trimmed[n_crossfade:]
            
            fade_curve = np.linspace(0, 1, n_crossfade)
            crossfade_bridge = (fade_out * (1 - fade_curve)) + (fade_in * fade_curve)
            
            output_chunks[-1] = prev_body
            output_chunks.append(crossfade_bridge)
            output_chunks.append(curr_body)

    return np.concatenate(output_chunks)

def split_text_into_chunks(
    text: str, 
    max_chars: int = 256
):
    """
    Chunks text. Determines flags based on how the *previous* chunk ended.
    Flags:
      0: Previous chunk ended with Sentence Terminator (. ? ! \n)
      1: Previous chunk was a hard cut (continuation)
      2: Previous chunk ended with Phrase Marker (, ; : —)
    """
    
    # Define sets for categorization
    SENTENCE_ENDERS = set('.!?…\n')
    PHRASE_MARKERS = set(',;:—')
    
    # 1. Regex split that captures the delimiters
    # Matches: One or more Sentence Enders OR One Phrase Marker
    # Note: We group them to keep the delimiter attached to the previous text or standalone
    parts = re.split(r'([.!?…\n]+|[,;:—])', text)
    
    # 2. Reassemble into atoms (Text + Punctuation)
    atoms = []
    
    # Iterate parts. Usually it's [text, delim, text, delim...]
    i = 0
    while i < len(parts):
        content = parts[i].strip()
        
        # If content is empty (e.g. text starts with punctuation), skip
        if not content:
            i += 1
            continue
            
        # Check if next part is a delimiter
        delim = ""
        if i + 1 < len(parts):
            # Check if it matches our delimiter sets
            candidate = parts[i+1]
            if any(c in SENTENCE_ENDERS for c in candidate) or any(c in PHRASE_MARKERS for c in candidate):
                delim = candidate
                i += 1 # Consume delimiter
        
        # Determine the type of this atom
        # Default to 1 (Continuation) if no delim found (shouldn't happen often with greedy regex, but safe fallback)
        atom_type = 1 
        
        if delim:
            if any(c in SENTENCE_ENDERS for c in delim):
                atom_type = 0
            elif any(c in PHRASE_MARKERS for c in delim):
                atom_type = 2
        
        atoms.append({
            'text': content + delim,
            'raw_text': content,
            'delim': delim,
            'type': atom_type # 0, 1, or 2
        })
        
        i += 1
    
    if not atoms:
        return {"chunks": [], "flags": []}

    # 3. Merge atoms into chunks
    final_chunks = []
    final_flags = []
    
    # State tracking
    current_chunk_str = ""
    # The flag for the *current* chunk being built.
    # The very first chunk always behaves like a new sentence (0).
    current_chunk_flag = 0 
    
    for atom in atoms:
        atom_text = atom['text']
        
        # Space logic: Add space if chunk not empty
        prefix = " " if current_chunk_str else ""
        cost = len(current_chunk_str) + len(prefix) + len(atom_text)
        
        if cost <= max_chars:
            # Fits in current chunk
            current_chunk_str += prefix + atom_text
            # The chunk's "ending type" is now determined by this atom
            # But we only use this if we finalize the chunk naturally
            last_atom_type = atom['type']
        else:
            # === OVERFLOW ===
            # The current atom does NOT fit.
            
            # 1. Finalize the PREVIOUS buffer
            if current_chunk_str:
                final_chunks.append(current_chunk_str)
                final_flags.append(current_chunk_flag)
                
                # The previous chunk ended naturally with whatever the last atom was.
                # So the NEXT chunk (which starts with current atom) will inherit that type?
                # NO. If we are here, it means we filled the previous chunk.
                # Logic:
                # If we are simply starting a new chunk because the atom didn't fit,
                # the previous chunk ended with its last atom's punctuation.
                current_chunk_flag = last_atom_type
            
            # 2. Check if current atom fits in a fresh chunk
            if len(atom_text) <= max_chars:
                current_chunk_str = atom_text
                last_atom_type = atom['type']
            else:
                # === HARD SPLIT ===
                # The atom itself is huge. We must split words.
                words = atom_text.split(' ')
                buffer = ""
                
                # If we just flushed, 'current_chunk_flag' is set correctly based on prev chunk end.
                # If this is the very first item, it is 0.
                
                first_part = True
                
                for word in words:
                    sp_needed = 1 if buffer else 0
                    if len(buffer) + sp_needed + len(word) + 1 <= max_chars:
                        buffer += (" " if buffer else "") + word
                    else:
                        # Flush buffer
                        # Assuming this is a hard cut, we add a period for TTS stability
                        # BUT logically, it is a Continuation (Flag 1) for the NEXT block.
                        final_chunks.append(buffer + ".") 
                        final_flags.append(current_chunk_flag if first_part else 1)
                        
                        buffer = word
                        first_part = False
                        
                        # Since we just sliced a word/sentence in half, the next chunk 
                        # MUST be a continuation.
                        current_chunk_flag = 1 
                
                # Remaining buffer becomes the start of the next process
                current_chunk_str = buffer
                # The type of this remainder is the original atom's type 
                # (e.g., if the huge sentence ended in '.', this remainder ends in '.')
                last_atom_type = atom['type']

    # 4. Final Flush
    if current_chunk_str:
        final_chunks.append(current_chunk_str)
        final_flags.append(current_chunk_flag)
        
    return {
        "chunks": final_chunks,
        "flags": final_flags
    }

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")