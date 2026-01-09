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

def punc_norm(text: str) -> str:
    """
    Refined Normalization for TTS:
    1. Standardizes quotes/dashes but preserves meaning.
    2. Keeps ? and ! for intonation.
    3. Handles spacing.
    """
    if not text or not text.strip():
        return ""

    # 1. Standardize quotes to straight quotes (easier to handle)
    # Replace smart quotes with straight quotes
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # 2. Capitalize first letter
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # 3. Filter specific "unspeakable" noise, BUT keep currency/math if your TTS supports it.
    # We allow: . , ? ! - $ % + =
    # We strip: ' " [ ] { } ( ) * < > / \ ^ # @ ~ ` _ |
    chars_to_remove = r'[\'\"\[\]\{\}\(\)\*\<\>\/\\\^\#\@\~\`\_\|]'
    text = re.sub(chars_to_remove, '', text)

    # 4. Normalize Pause Markers
    # Map :, ;, — (em-dash) to Comma. 
    # NOTE: We do NOT map standard hyphen (-) to comma to protect words like "re-do"
    text = re.sub(r'[:;—]', ',', text)
    
    # 5. Normalize Ellipsis -> Period (or comma if you prefer short pause)
    text = re.sub(r'[…]', '.', text)

    # 6. Clean up spacing around punctuation
    # "word ," -> "word,"
    text = re.sub(r'\s+([,.?!])', r'\1', text)
    # "word,word" -> "word, word" (Add space after punctuation if missing)
    text = re.sub(r'([,.?!])(?=[a-zA-Z0-9])', r'\1 ', text)

    # 7. Deduplication
    text = re.sub(r'\.+', '.', text)     # ...... -> .
    text = re.sub(r'!+', '!', text)      # !!!!!! -> !
    text = re.sub(r'\?+', '?', text)     # ?????? -> ?
    text = re.sub(r',+', ',', text)      # ,,,,,, -> ,
    
    # Resolve punctuation conflicts (Period wins over comma)
    text = re.sub(r'[,]+\.', '.', text)
    text = re.sub(r'\.[,]+', '.', text)

    # 8. Collapse whitespace
    text = " ".join(text.split())

    # 9. Ensure valid ending
    if text and text[-1] not in '.?!':
        if text.endswith(','):
            text = text[:-1] + "."
        else:
            text += "."

    return text

def split_text_into_chunks(text: str, max_chars: int = 256):
    """
    Chunks text preserving semantic boundaries and intonation markers.
    """
    if not text:
        return {"chunks": [], "flags": []}
        
    # Define sets
    # We keep ? and ! as sentence enders to preserve intonation in the chunk
    SENTENCE_ENDERS = set('.!?\n') 
    PHRASE_MARKERS = set(',')

    # 1. Regex split (Keep delimiters)
    # Split by . ? ! or ,
    parts = re.split(r'([.!?\n]+|[,])', text)
    
    atoms = []
    i = 0
    while i < len(parts):
        content = parts[i].strip()
        if not content:
            i += 1
            continue
            
        delim = ""
        # Check if next part is a delimiter
        if i + 1 < len(parts):
            candidate = parts[i+1]
            if any(c in SENTENCE_ENDERS or c in PHRASE_MARKERS for c in candidate):
                delim = candidate
                i += 1
        
        # Determine Atom Type (for pause flags)
        # 0 = Sentence End (Long pause)
        # 2 = Phrase End (Short pause / Comma)
        # 1 = Continuation (No delim)
        atom_type = 1 
        
        if delim:
            if any(c in SENTENCE_ENDERS for c in delim):
                atom_type = 0
            elif any(c in PHRASE_MARKERS for c in delim):
                atom_type = 2
        
        atoms.append({
            'text': content + delim,
            'type': atom_type
        })
        i += 1
    
    if not atoms:
        return {"chunks": [], "flags": []}

    # 2. Merge logic
    final_chunks = []
    final_flags = []
    
    current_chunk_str = ""
    current_chunk_flag = 0 
    
    for atom in atoms:
        atom_text = atom['text']
        
        prefix = " " if current_chunk_str else ""
        cost = len(current_chunk_str) + len(prefix) + len(atom_text)
        
        if cost <= max_chars:
            current_chunk_str += prefix + atom_text
            # Identify the flag based on the *last* atom added to this chunk
            last_atom_type = atom['type'] 
        else:
            # === OVERFLOW ===
            # Push current buffer
            if current_chunk_str:
                # Ensure the chunk ends with a "stopper" for TTS stability
                # If it ended with a comma, we swap to period for the audio file 
                # (so pitch drops), but keep flag 2 if you want shorter silence logic elsewhere.
                # However, usually for TTS chunks:
                # Ending with ',' often leaves the model "hanging" (pitch stays up).
                # It is safer to force '.' at the end of a physical chunk.
                final_text = current_chunk_str
                if final_text.strip()[-1] == ',':
                     final_text = final_text.strip()[:-1] + "."

                final_chunks.append(final_text)
                final_flags.append(current_chunk_flag)
                
                # The flag for the NEXT chunk is determined by how the PREVIOUS atom ended.
                # If previous atom was type 0 (.), this new chunk starts fresh (flag 0).
                # Actually, usually flags represent the PAUSE AFTER the chunk.
                current_chunk_flag = last_atom_type

            # Check if new atom fits
            if len(atom_text) <= max_chars:
                current_chunk_str = atom_text
                last_atom_type = atom['type']
            else:
                # === HARD SPLIT (Word by Word) ===
                words = atom_text.split(' ')
                buffer = ""
                
                for word in words:
                    sp = " " if buffer else ""
                    if len(buffer) + len(sp) + len(word) + 1 <= max_chars:
                        buffer += sp + word
                    else:
                        # Flush hard chunk
                        # We append "..." or "." to indicate a break
                        final_chunks.append(buffer + "...") 
                        final_flags.append(1) # Flag 1 = Continuation (very short/no pause)
                        buffer = word
                
                current_chunk_str = buffer
                last_atom_type = atom['type']

    # Final flush
    if current_chunk_str:
        final_text = current_chunk_str
        # Fix trailing comma
        if final_text.strip()[-1] == ',':
             final_text = final_text.strip()[:-1] + "."
             
        final_chunks.append(final_text)
        # The final chunk usually gets a full stop pause (0)
        final_flags.append(0)

    return {
        "chunks": final_chunks,
        "flags": final_flags
    }

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")