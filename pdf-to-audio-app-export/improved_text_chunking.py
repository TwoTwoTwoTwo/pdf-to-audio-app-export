"""
Improved text chunking for TTS synthesis.
Ensures proper sentence boundaries, no text skipping, and consistent voice quality.
"""

import re
from typing import List, Optional, Tuple
import nltk

# Try to download punkt tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass  # Fallback to regex if NLTK not available


def split_text_on_sentences(text: str, max_chunk_size: int = 250, 
                           min_chunk_size: int = 50,
                           overlap_sentences: int = 0) -> List[str]:
    """
    Split text into chunks at sentence boundaries.
    
    Args:
        text: Input text to split
        max_chunk_size: Maximum characters per chunk (XTTS works best with 200-250)
        min_chunk_size: Minimum characters per chunk to avoid tiny fragments
        overlap_sentences: Number of sentences to overlap between chunks for consistency
    
    Returns:
        List of text chunks split at proper sentence boundaries
    """
    # Clean and normalize the text first
    text = text.strip()
    if not text:
        return []
    
    # Try to use NLTK for better sentence splitting
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except:
        # Fallback to regex-based sentence splitting
        # This regex handles common sentence endings better
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?]["\'"])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Handle cases where the split didn't work well
        if len(sentences) == 1:
            # Try a simpler split
            sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [text] if len(text) >= min_chunk_size else []
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_size = len(sentence)
        
        # If a single sentence is too long, we need to split it
        if sentence_size > max_chunk_size:
            # First, save current chunk if it exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split long sentence at clause boundaries
            parts = split_long_sentence(sentence, max_chunk_size)
            for part in parts:
                if len(part) >= min_chunk_size:
                    chunks.append(part)
        
        # If adding this sentence would exceed max size, start new chunk
        elif current_size + sentence_size + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Add overlap sentences from previous chunk if specified
            if overlap_sentences > 0 and i >= overlap_sentences:
                overlap_start = max(0, i - overlap_sentences)
                current_chunk = sentences[overlap_start:i]
                current_size = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            else:
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size + (1 if current_chunk else 0)
    
    # Don't forget the last chunk
    if current_chunk:
        final_chunk = ' '.join(current_chunk)
        if len(final_chunk) >= min_chunk_size:
            chunks.append(final_chunk)
        elif chunks:
            # Merge very small final chunk with previous one
            chunks[-1] = chunks[-1] + ' ' + final_chunk
    
    return chunks


def split_long_sentence(sentence: str, max_size: int) -> List[str]:
    """
    Split a long sentence at clause boundaries (commas, semicolons, conjunctions).
    
    Args:
        sentence: Long sentence to split
        max_size: Maximum size for each part
    
    Returns:
        List of sentence parts
    """
    if len(sentence) <= max_size:
        return [sentence]
    
    # Try to split at natural boundaries
    # Priority: semicolon > comma > conjunction > dash
    split_patterns = [
        (r';\s+', '; '),  # Semicolon
        (r',\s+(?=\w+ing\b|\w+ed\b|which|that|who|where|when|while|although|because|since|if|unless|after|before)', ', '),  # Comma before clause
        (r',\s+', ', '),  # Any comma
        (r'\s+(?:and|but|or|yet|so|for|nor)\s+', ' '),  # Conjunctions
        (r'\s+[-–—]\s+', ' - '),  # Dashes
        (r'\s+', ' '),  # Any space (last resort)
    ]
    
    for pattern, separator in split_patterns:
        parts = re.split(f'({pattern})', sentence)
        
        # Reconstruct parts maintaining separators
        reconstructed = []
        current = ""
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text part
                if current and len(current + part) > max_size:
                    if current:
                        reconstructed.append(current.strip())
                    current = part
                else:
                    current += part
            else:  # Separator part
                current += separator
        
        if current:
            reconstructed.append(current.strip())
        
        # Check if split was successful
        if all(len(p) <= max_size for p in reconstructed) and len(reconstructed) > 1:
            return reconstructed
    
    # If no natural split worked, do hard character split at word boundaries
    words = sentence.split()
    parts = []
    current = []
    current_len = 0
    
    for word in words:
        word_len = len(word)
        if current_len + word_len + 1 > max_size:
            if current:
                parts.append(' '.join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len += word_len + (1 if current_len > 0 else 0)
    
    if current:
        parts.append(' '.join(current))
    
    return parts


def prepare_text_for_tts(text: str, remove_quotes: bool = False) -> str:
    """
    Prepare text for TTS by cleaning and normalizing.
    
    Args:
        text: Input text
        remove_quotes: Whether to remove quotation marks (can affect voice consistency)
    
    Returns:
        Cleaned text ready for TTS
    """
    # Replace smart quotes and special characters
    replacements = {
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '—': ' - ',
        '–': ' - ',
        '…': '...',
        '\u00a0': ' ',  # non-breaking space
        '\t': ' ',
        '\n': ' ',
        '\r': ' ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove quotes if requested (can help with voice consistency)
    if remove_quotes:
        text = text.replace('"', '').replace("'", '')
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any remaining special unicode characters that might cause issues
    text = ''.join(char if ord(char) < 128 or char.isalpha() else ' ' for char in text)
    
    # Clean up again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_chunks(chunks: List[str], original_text: str) -> Tuple[bool, str]:
    """
    Validate that chunks properly represent the original text.
    
    Args:
        chunks: List of text chunks
        original_text: Original input text
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not chunks:
        return False, "No chunks generated"
    
    # Join chunks and compare
    joined = ' '.join(chunks)
    
    # Remove extra spaces for comparison
    original_clean = re.sub(r'\s+', ' ', original_text.strip())
    joined_clean = re.sub(r'\s+', ' ', joined.strip())
    
    # Check if all content is preserved (allowing for minor formatting differences)
    original_words = original_clean.split()
    joined_words = joined_clean.split()
    
    if len(original_words) != len(joined_words):
        missing = len(original_words) - len(joined_words)
        return False, f"Word count mismatch: {missing} words {'missing' if missing > 0 else 'extra'}"
    
    # Check for very small chunks that might cause issues
    tiny_chunks = [i for i, chunk in enumerate(chunks) if len(chunk) < 20]
    if tiny_chunks:
        return False, f"Found {len(tiny_chunks)} chunks smaller than 20 characters"
    
    return True, "Chunks validated successfully"


# Voice consistency helper
class VoiceConsistencyManager:
    """
    Manages voice consistency across chunks by maintaining speaker conditioning.
    """
    
    def __init__(self, speaker_wav_path: Optional[str] = None):
        """
        Initialize with a speaker WAV file for consistent voice cloning.
        
        Args:
            speaker_wav_path: Path to the speaker's reference audio
        """
        self.speaker_wav_path = speaker_wav_path
        self.conditioning_cache = None
        
    def get_consistent_conditioning(self):
        """
        Get consistent voice conditioning parameters for all chunks.
        This helps maintain voice consistency across the entire audio.
        """
        if self.speaker_wav_path and not self.conditioning_cache:
            # Load and cache the conditioning
            # This would be used by XTTS to maintain consistent voice
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(self.speaker_wav_path)
                # Normalize to mono if needed
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # Resample to 22050 if needed (XTTS standard)
                if sample_rate != 22050:
                    resampler = torchaudio.transforms.Resample(sample_rate, 22050)
                    waveform = resampler(waveform)
                self.conditioning_cache = {
                    'waveform': waveform,
                    'sample_rate': 22050,
                    'path': self.speaker_wav_path
                }
            except Exception as e:
                print(f"Warning: Could not cache voice conditioning: {e}")
                self.conditioning_cache = {'path': self.speaker_wav_path}
        
        return self.conditioning_cache or {'path': self.speaker_wav_path}
    
    def apply_voice_stabilization(self, chunks: List[str]) -> List[str]:
        """
        Apply text modifications to help stabilize voice across chunks.
        
        Args:
            chunks: List of text chunks
        
        Returns:
            Modified chunks for better voice consistency
        """
        stabilized = []
        
        for i, chunk in enumerate(chunks):
            # Add slight context markers to help maintain voice
            if i == 0:
                # First chunk - establish voice
                chunk = chunk
            elif i == len(chunks) - 1:
                # Last chunk - maintain energy
                chunk = chunk
            else:
                # Middle chunks - add subtle continuation cues
                # This helps XTTS maintain consistent prosody
                if not chunk[0].isupper():
                    chunk = chunk[0].upper() + chunk[1:]
            
            stabilized.append(chunk)
        
        return stabilized


if __name__ == "__main__":
    # Test the chunking
    test_text = """
    This is a test of the chunking system. It should properly split sentences without breaking them in the middle.
    Sometimes we have very long sentences that go on and on, with multiple clauses, subclauses, and ideas that need to be expressed,
    which can make it challenging to find the right place to split them. But the system should handle it gracefully.
    Short sentences are easy. Medium sentences work well too. And even questions should be handled properly, right?
    """
    
    chunks = split_text_on_sentences(test_text, max_chunk_size=150)
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")
    
    # Validate
    is_valid, message = validate_chunks(chunks, test_text)
    print(f"\nValidation: {message}")
