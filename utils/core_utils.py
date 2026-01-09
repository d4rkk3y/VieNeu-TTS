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
    Chunks text using a strict hierarchy:
    1. Split by Sentence terminators.
    2. If a sentence > max_chars, split that specific sentence by Phrase markers.
    3. If a phrase > max_chars, split that specific phrase by Words.
    
    Constraint: Never merge a whole sentence with a fragment of a split sentence.
    """
    if not text:
        return {"chunks": [], "flags": []}

    # --- Definitions ---
    # 0 = Sentence End (Long pause)
    # 2 = Phrase End (Short pause)
    # 1 = Continuation (No pause / jagged split)
    SENTENCE_ENDERS = set('.!?\n') 
    PHRASE_MARKERS = set(',;:') # Added ; and : for robustness

    final_chunks = []
    final_flags = []

    def get_flag(chunk_text):
        """Determine pause flag based on the last character."""
        s = chunk_text.strip()
        if not s: return 1
        last_char = s[-1]
        if last_char in SENTENCE_ENDERS:
            return 0
        if last_char in PHRASE_MARKERS:
            return 2
        return 1

    def clean_and_finalize_chunk(txt):
        """Helper to ensure TTS stability (e.g. converting trailing , to .)."""
        txt = txt.strip()
        if not txt: return
        
        flag = get_flag(txt)
        
        # If a chunk ends in a comma but is being isolated, 
        # it is often better for TTS to treat it as a period to drop pitch.
        # However, we preserve the flag '2' if you handle shorter pauses downstream,
        # OR we can force it to '.'. 
        # Here we follow the standard stability practice:
        if txt[-1] == ',':
            txt = txt[:-1] + "."
        
        final_chunks.append(txt)
        final_flags.append(flag)

    def split_with_delimiters(text, pattern):
        """Splits text but attaches the delimiter to the preceding text."""
        parts = re.split(pattern, text)
        result = []
        i = 0
        while i < len(parts):
            content = parts[i]
            delim = ""
            if i + 1 < len(parts):
                delim = parts[i+1]
            
            full_atom = content + delim
            if full_atom.strip():
                result.append(full_atom)
            i += 2
        return result

    # ==========================================
    # LEVEL 1: Split by Sentences
    # ==========================================
    sentences = split_with_delimiters(text, r'([.!?\n]+)')
    
    current_buffer = ""

    for sentence in sentences:
        
        # Check if this single sentence exceeds the limit
        if len(sentence) > max_chars:
            # === CRITICAL REQUIREMENT ===
            # "Do not merge a sentence with a part of sentence which split by lower priority"
            # So, flush whatever whole sentences we have accumulated so far.
            if current_buffer:
                clean_and_finalize_chunk(current_buffer)
                current_buffer = ""

            # Now process the huge sentence individually (Level 2)
            
            # ==========================================
            # LEVEL 2: Split by Phrases (within the huge sentence)
            # ==========================================
            phrases = split_with_delimiters(sentence, r'([,;:]+)')
            phrase_buffer = ""

            for phrase in phrases:
                if len(phrase) > max_chars:
                    # Flush existing phrase buffer
                    if phrase_buffer:
                        clean_and_finalize_chunk(phrase_buffer)
                        phrase_buffer = ""
                    
                    # ==========================================
                    # LEVEL 3: Split by Words (within the huge phrase)
                    # ==========================================
                    words = phrase.split(' ')
                    word_buffer = ""
                    
                    for word in words:
                        sp = " " if word_buffer else ""
                        if len(word_buffer) + len(sp) + len(word) + 1 <= max_chars:
                            word_buffer += sp + word
                        else:
                            # Hard split overflow
                            clean_and_finalize_chunk(word_buffer + "...")
                            word_buffer = word # Start new with current word
                    
                    if word_buffer:
                        # Append the remainder of the word split. 
                        # Note: This remainder carries the original punctuation of the phrase.
                        clean_and_finalize_chunk(word_buffer)

                else:
                    # Phrase fits
                    sp = " " if phrase_buffer else ""
                    if len(phrase_buffer) + len(sp) + len(phrase) <= max_chars:
                        phrase_buffer += sp + phrase
                    else:
                        clean_and_finalize_chunk(phrase_buffer)
                        phrase_buffer = phrase
            
            # Flush remaining phrases from this specific huge sentence
            if phrase_buffer:
                clean_and_finalize_chunk(phrase_buffer)

        else:
            # Sentence fits comfortably. 
            # Standard merge logic with previous WHOLE sentences.
            sp = " " if current_buffer else ""
            if len(current_buffer) + len(sp) + len(sentence) <= max_chars:
                current_buffer += sp + sentence
            else:
                clean_and_finalize_chunk(current_buffer)
                current_buffer = sentence

    # Final flush of any whole sentences left
    if current_buffer:
        clean_and_finalize_chunk(current_buffer)

    return {
        "chunks": final_chunks,
        "flags": final_flags
    }

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")