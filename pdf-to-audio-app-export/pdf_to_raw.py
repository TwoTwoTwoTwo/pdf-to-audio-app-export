import pdfplumber
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import re
import os
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # Make language detection deterministic

def detect_multilingual_content(full_text: str) -> Tuple[str, Dict[str, float]]:
    """Detect multiple languages in text content with more comprehensive analysis"""
    from collections import Counter
    import string
    
    # Primary language detection using langdetect on full text
    try:
        primary_language = detect(full_text)
    except Exception:
        primary_language = "unknown"
    
    # Split text into chunks for better language detection
    sentences = full_text.split('. ')
    chunk_size = 50  # sentences per chunk
    chunks = ['. '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    
    language_counts = Counter()
    total_chunks = len(chunks)
    
    # Detect language for each chunk
    for chunk in chunks:
        if len(chunk.strip()) < 50:  # Skip very short chunks
            continue
        try:
            chunk_lang = detect(chunk)
            language_counts[chunk_lang] += 1
        except Exception:
            continue
    
    # Calculate language proportions
    language_proportions = {}
    for lang, count in language_counts.items():
        language_proportions[lang] = count / total_chunks if total_chunks > 0 else 0
    
    # Additional language detection based on word frequency
    text_lower = full_text.lower()
    
    # Common words in different languages for additional detection
    language_markers = {
        'ms': ['yang', 'dan', 'dengan', 'adalah', 'untuk', 'pada', 'dari', 'dalam', 'akan', 'sebagai',
               'saya', 'kita', 'mereka', 'negara', 'kerajaan', 'malaysia', 'singapura', 'melayu'],
        'zh': ['的', '在', '和', '是', '了', '有', '我', '他', '这', '个', '中国', '一个', '不是'],
        'ta': ['மற்றும்', 'இந்த', 'ஒரு', 'அந்த', 'இல்', 'என்', 'இருந்தது', 'சிங்கப்பூர்'],
        'hi': ['और', 'का', 'एक', 'में', 'है', 'के', 'की', 'से', 'पर', 'को']
    }
    
    word_count_analysis = {}
    words = [word.strip(string.punctuation) for word in text_lower.split()]
    total_words = len(words)
    
    for lang, markers in language_markers.items():
        marker_count = sum(1 for word in words if word in markers)
        if total_words > 0:
            word_count_analysis[lang] = marker_count / total_words
        else:
            word_count_analysis[lang] = 0
    
    # Combine detection methods
    combined_analysis = {}
    
    # Start with chunk-based proportions
    for lang in language_proportions:
        combined_analysis[lang] = language_proportions[lang]
    
    # Add word-based analysis with lower weight
    for lang, proportion in word_count_analysis.items():
        if proportion > 0.001:  # Only include if significant presence (>0.1%)
            if lang in combined_analysis:
                combined_analysis[lang] += proportion * 0.3  # Lower weight for word analysis
            else:
                combined_analysis[lang] = proportion * 0.3
    
    # Ensure primary language is included
    if primary_language not in combined_analysis:
        combined_analysis[primary_language] = 0.5
    
    # Sort by proportion and format results
    sorted_languages = sorted(combined_analysis.items(), key=lambda x: x[1], reverse=True)
    
    # Create readable language info
    language_info = {}
    language_names = {
        'en': 'English',
        'ms': 'Malay',
        'zh': 'Chinese', 
        'zh-cn': 'Chinese',
        'ta': 'Tamil',
        'hi': 'Hindi',
        'af': 'Afrikaans',
        'unknown': 'Unknown'
    }
    
    for lang, proportion in sorted_languages[:5]:  # Top 5 languages
        if proportion > 0.005:  # Only show if >0.5% presence
            lang_name = language_names.get(lang, lang.capitalize())
            language_info[lang_name] = round(proportion * 100, 2)
    
    # Format primary language result
    if sorted_languages:
        top_lang = sorted_languages[0][0]
        top_lang_name = language_names.get(top_lang, top_lang.capitalize())
    else:
        top_lang_name = language_names.get(primary_language, primary_language.capitalize())
    
    return top_lang_name, language_info


def ocr_page_fitz(page: fitz.Page) -> str:
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img, lang='eng')
    return text


def ocr_page_pdfplumber(page) -> str:
    pil_image = page.to_image(resolution=300).original
    text = pytesseract.image_to_string(pil_image, lang='eng')
    return text


def extract_metadata(pdf) -> Tuple[Optional[str], Optional[str]]:
    """Extract title and author from multiple sources"""
    # Try PDF metadata first
    metadata = pdf.metadata or {}
    title = metadata.get('Title')
    author = metadata.get('Author')
    
    # Clean up metadata if found
    if title and title.strip() and not title.strip().lower() in ['untitled', 'document', '']:
        title = title.strip()
    else:
        title = None
        
    if author and author.strip():
        author = author.strip()
    else:
        author = None
    
    # If no good title from metadata, try to extract from first few pages
    if not title:
        title = extract_title_from_pages(pdf)
    
    # If no author from metadata, try to extract from first few pages  
    if not author:
        author = extract_author_from_pages(pdf)
        
    return title or "Unknown Title", author or "Unknown Author"

def extract_title_from_pages(pdf) -> Optional[str]:
    """Extract likely title from first few pages"""
    potential_titles = []
    
    # Check first 5 pages for title patterns
    for page_num in range(min(5, len(pdf.pages))):
        page = pdf.pages[page_num]
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        if not lines:
            continue
        
        # Look for common title patterns
        full_text = " ".join(lines)
        
        # Pattern 1: "The [Title] Story" or "The [Title]"
        title_patterns = [
            r'The\s+([A-Z][a-zA-Z\s]+)\s+Story',
            r'The\s+([A-Z][a-zA-Z\s]{5,30})',
            r'Memoirs?\s+of\s+([A-Z][a-zA-Z\s\.]{5,40})',
            r'^([A-Z][a-zA-Z\s]{10,50})$',  # Title case line
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                title = match.strip()
                # Filter out unwanted matches
                if (len(title) >= 5 and 
                    not re.match(r'^\d+', title) and
                    'chapter' not in title.lower() and
                    'page' not in title.lower() and
                    not title.lower().startswith('by')):
                    potential_titles.append((title, page_num, 0))
                    
        # Look for title patterns in first few lines
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            
            # Skip very short or very long lines
            if len(line) < 8 or len(line) > 100:
                continue
                
            # Skip lines that look like headers, footers, or page numbers
            if re.match(r'^(page|chapter|\d+|©|copyright|isbn|by\s+)', line.lower()):
                continue
                
            # Skip lines with too many special characters or numbers
            if (len(re.findall(r'[^a-zA-Z\s]', line)) > len(line) * 0.4 or
                len(re.findall(r'\d', line)) > len(line) * 0.2):
                continue
                
            # Look for likely book titles
            if (10 <= len(line) <= 80 and 
                any(word[0].isupper() for word in line.split() if word) and
                not line.endswith('.') and
                line.count(' ') >= 1 and  # At least 2 words
                line.count(' ') <= 8):    # But not too many
                
                # Give higher priority to certain patterns
                priority = 10 - i  # Earlier lines get higher priority
                if 'story' in line.lower():
                    priority += 10
                if 'memoir' in line.lower():
                    priority += 8
                if line.isupper() or line.istitle():
                    priority += 5
                    
                potential_titles.append((line, page_num, priority))
    
    if potential_titles:
        # Sort by priority (descending), then page number, then line position
        potential_titles.sort(key=lambda x: (-x[2], x[1]))
        return potential_titles[0][0]
        
    return None

def extract_author_from_pages(pdf) -> Optional[str]:
    """Extract likely author from first few pages"""
    author_patterns = [
        r'by\s+([a-zA-Z\s\.]+)',
        r'author[:\s]+([a-zA-Z\s\.]+)',
        r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$',  # Name pattern
    ]
    
    for page_num in range(min(3, len(pdf.pages))):
        page = pdf.pages[page_num]
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        for line in lines[:15]:  # Check first 15 lines of each page
            for pattern in author_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    author = match.group(1).strip()
                    # Clean up author name
                    if 5 <= len(author) <= 50 and not re.search(r'\d', author):
                        return author
                        
    return None


def clean_toc_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.replace('–', '-').replace('—', '-').replace('‒', '-')
    toc_pattern = re.compile(r"""
        ^\s*
        (?P<title>.+?)               
        [\.\-\s]{2,}                
        (?P<page>\S+)\s*$           
        """, re.VERBOSE)
    match = toc_pattern.match(line)
    if not match:
        return None
    title = match.group('title').strip()
    title = re.sub(r'[\.\-\s\–\—\_]+$', '', title)
    page = match.group('page').strip()
    return title, page


def roman_to_int(s: str) -> Optional[int]:
    roman_map = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
    s = s.lower()
    if not all(ch in roman_map for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        val = roman_map[ch]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total if total > 0 else None


def printed_page_to_pdf_index(printed_page: str, mapping: Dict[str, int]) -> Optional[int]:
    if printed_page in mapping:
        return mapping[printed_page]
    try:
        num = int(printed_page)
        numeric_keys = []
        for k in mapping:
            try:
                nk = int(k)
                numeric_keys.append((nk, mapping[k]))
            except Exception:
                continue
        candidates = [idx for nk, idx in numeric_keys if nk >= num]
        if candidates:
            return min(candidates)
    except Exception:
        pass
    val = roman_to_int(printed_page)
    if val is not None:
        for k, idx in mapping.items():
            if roman_to_int(k) == val:
                return idx
    return None


def extract_printed_page_numbers(pdf) -> Dict[str, int]:
    printed_to_pdf_idx = {}
    page_num_pattern = re.compile(r"^(?:page\s*)?([ivxlcdm\d]+)$", re.I)
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        footer_lines = lines[-2:]
        found_number = None
        for line in footer_lines:
            m = page_num_pattern.match(line.lower().replace(".", "").strip())
            if m:
                found_number = m.group(1)
                break
        if not found_number:
            line = footer_lines[-1]
            if re.fullmatch(r"[ivxlcdm\d]+", line.lower()):
                found_number = line.lower()
        if found_number and found_number not in printed_to_pdf_idx:
            printed_to_pdf_idx[found_number] = i
    return printed_to_pdf_idx


def is_page_text_low_quality(text: str) -> bool:
    if not text or len(text.strip()) < 100:
        return True
    return False


def is_blank_or_image_only_page(page) -> bool:
    text = page.extract_text() or ""
    has_text = bool(text.strip())
    has_images = bool(page.images)
    return not has_text and not has_images


def extract_toc_candidates(pdf, printed_page_map: Dict[str, int]) -> List[Tuple[str, int, str]]:
    toc = []
    seen_titles_pages = set()
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if is_page_text_low_quality(text):
            text = ocr_page_pdfplumber(page)
        new_entries = 0
        for line in text.splitlines():
            cleaned = clean_toc_line(line.strip())
            if cleaned:
                title, printed_page_str = cleaned
                pdf_idx = printed_page_to_pdf_index(printed_page_str, printed_page_map)
                if pdf_idx is not None and (title, pdf_idx) not in seen_titles_pages:
                    toc.append((title, pdf_idx, printed_page_str))
                    seen_titles_pages.add((title, pdf_idx))
                    new_entries += 1
        if i > 50 and new_entries == 0:
            break
    toc.sort(key=lambda x: x[1])
    return toc


def extract_toc_from_bookmarks(doc: fitz.Document) -> List[Tuple[str, int, Optional[str]]]:
    toc = []
    try:
        outline = doc.get_toc()
        for level, title, page_num in outline:
            toc.append((title.strip(), page_num - 1, None))
    except Exception:
        pass
    return toc


def find_chapter_starts_fitz(doc: fitz.Document, font_size_threshold=14) -> List[int]:
    chapter_pages = []
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            max_font_size = 0
            for b in blocks:
                if b['type'] != 0:
                    continue
                for line in b["lines"]:
                    for span in line["spans"]:
                        size = span["size"]
                        if size > max_font_size:
                            max_font_size = size
            if max_font_size >= font_size_threshold:
                chapter_pages.append(page_num)
        except Exception:
            continue
    return sorted(set(chapter_pages))


def is_rear_matter_title(title: str) -> bool:
    keywords = [
        'index', 'glossary', 'chronology', 'appendix', 'bibliography', 'references', 'notes',
        'colophon', 'about the author', 'afterword', 'acknowledgements'
    ]
    title_lower = title.lower()
    return any(kw in title_lower for kw in keywords)


def save_text_file(path: Path, text: str):
    path.write_text(text.strip() + "\n", encoding="utf-8")


def fix_toc_title(title: str) -> str:
    # Remove leading chapter numbers like "33. Konfrontasi" -> "Konfrontasi"
    cleaned = re.sub(r"^\d+[\.\)]\s*", "", title)
    return cleaned.strip()


def format_toc_entries(entries: List[Tuple[str, int, Optional[str]]], section_num: int) -> List[str]:
    lines = []
    for idx, (t, p, _) in enumerate(entries, start=1):
        cleaned_title = fix_toc_title(t)
        lines.append(f"{section_num}.{idx} {cleaned_title}  → page {p+1}")
    return lines


def postprocess_toc(
    toc: List[Tuple[str, int, Optional[str]]],
    chapter_starts: List[int],
    margin_page_numbers: List[int],
    page_tolerance: int = 2,
) -> List[Tuple[str, int, Optional[str]]]:
    seen_titles = set()
    seen_chapter_nums = set()
    cleaned = []
    for entry in toc:
        title = entry[0].strip()
        page = entry[1]
        # Extract chapter number if present at start, e.g., "2.9" or "33."
        m = re.match(r"^\s*(\d+(?:\.\d+)?)", title)
        chapter_num = m.group(1) if m else None

        # Normalize title to lower for duplicate detection
        title_lower = title.lower()

        # Skip if title or chapter number already seen
        if title_lower in seen_titles:
            continue
        if chapter_num and chapter_num in seen_chapter_nums:
            continue

        cleaned.append(entry)
        seen_titles.add(title_lower)
        if chapter_num:
            seen_chapter_nums.add(chapter_num)

    cleaned.sort(key=lambda x: x[1])

    # Additionally, remove entries that start on the same page except first
    filtered = []
    last_page = -1
    for entry in cleaned:
        if entry[1] != last_page:
            filtered.append(entry)
            last_page = entry[1]

    return filtered



def detect_missing_chapters(toc: List[Tuple[str, int, Optional[str]]]) -> List[int]:
    chapter_nums = []
    for title, _, _ in toc:
        m = re.match(r"^\s*(\d+)", title)
        if m:
            chapter_nums.append(int(m.group(1)))
    chapter_nums = sorted(set(chapter_nums))
    missing = []
    for i in range(chapter_nums[0], chapter_nums[-1]):
        if i not in chapter_nums:
            missing.append(i)
    return missing


def clean_page_text(text: str, page_num: int) -> str:
    """Clean page text by removing marginalia, page numbers, and chapter titles"""
    lines = text.splitlines()
    cleaned_lines = []
    
    # Enhanced marginalia patterns with more comprehensive coverage
    marginalia_patterns = [
        # Page numbers in various formats
        r'^\d{1,4}$',  # Standalone page numbers
        r'^\d{1,4}\s*$',  # Page numbers with trailing spaces
        r'^\s*\d{1,4}\s*$',  # Page numbers with leading/trailing spaces
        r'^\d{1,4}\s*[-–—]\s*$',  # Page numbers with dashes
        r'^[-–—]\s*\d{1,4}\s*[-–—]$',  # Page numbers surrounded by dashes
        
        # Roman numerals
        r'^[ivxlcdm]{1,7}$',  # Roman numerals
        r'^[IVXLCDM]{1,7}$',  # Uppercase Roman numerals
        
        # Chapter and page headers
        r'^(page|chapter|ch\.|chap)\s*\d+$',  # Page/chapter headers
        r'^\d+\s+(chapter|ch\.|chap|page)\s*\d*$',  # Page + chapter
        r'^chapter\s+[ivxlcdm]+$',  # Chapter with roman numerals
        r'^chapter\s+[IVXLCDM]+$',  # Chapter with uppercase roman
        
        # Book and chapter titles (more generic patterns)
        r'^[A-Z][a-z]+\s+Story$',  # "Something Story" titles
        r'^The\s+[A-Z][a-z]+\s+Story$',  # "The Something Story" titles
        r'^[A-Z][a-zA-Z\s]+,\s+Independence$',  # "Something, Independence"
        r'^Growing\s+Up$',
        r'^The\s+Japanese\s+Invaders?$',
        r'^After\s+the\s+Liberation$',
        r'^My\s+[A-Z][a-zA-Z\s]+\s+Days$',  # "My Something Days"
        r'^Work,\s+Wedding\s+and\s+Politics$',
        
        # Generic repeated titles (check for all caps or title case patterns)
        r'^[A-Z][A-Z\s]{10,50}$',  # ALL CAPS titles
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,8}$',  # Title Case repeated patterns
        
        # Separator lines and artifacts
        r'^[\-_=]{3,}$',  # Separator lines
        r'^\s*[|\-_=•]{2,}\s*$',  # Various separator patterns
        r'^\s*(left|right|verso|recto)\s*$',  # Page side indicators
        r'^\s*\*{2,}\s*$',  # Asterisk separators
        r'^\s*\#{2,}\s*$',  # Hash separators
        
        # Copyright and publication info
        r'^©.*$',  # Copyright lines
        r'^copyright.*$',  # Copyright text
        r'^\d{4}.*copyright.*$',  # Year + copyright
        r'^ISBN.*$',  # ISBN lines
        r'^published.*$',  # Publication info
        
        # Page positioning indicators
        r'^\d+\s+of\s+\d+$',  # "Page X of Y"
        r'^page\s+\d+\s+of\s+\d+$',  # "Page X of Y" with text
        
        # Short isolated lines that are likely artifacts
        r'^[a-zA-Z]$',  # Single letters
        r'^[a-zA-Z]{1,2}\d*$',  # 1-2 letters possibly with numbers
        r'^\d+[a-zA-Z]{1,2}$',  # Numbers with 1-2 letters
    ]
    
    # Track potential repeated chapter titles by frequency
    line_frequency = {}
    for line in lines:
        clean_line = line.strip()
        if clean_line and 5 < len(clean_line) < 100:  # Reasonable title length
            line_frequency[clean_line] = line_frequency.get(clean_line, 0) + 1
    
    # Identify lines that appear multiple times (likely repeated titles)
    repeated_lines = {line for line, count in line_frequency.items() if count >= 2}
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        if not line:
            cleaned_lines.append("")  # Keep paragraph spacing
            continue
        
        # Check if line is marginalia
        is_marginalia = any(re.match(pattern, line, re.IGNORECASE) for pattern in marginalia_patterns)
        if is_marginalia:
            continue
            
        # Skip repeated chapter titles or book titles
        if line in repeated_lines and len(line) > 10:
            continue
            
        # Skip lines that are mostly punctuation or numbers
        alpha_chars = sum(1 for c in line if c.isalpha())
        if len(line) > 5 and alpha_chars / len(line) < 0.3:
            continue
            
        # Skip very short lines at the beginning or end (likely artifacts)
        line_pos = lines.index(original_line)
        is_near_boundary = line_pos < 3 or line_pos >= len(lines) - 3
        if is_near_boundary and len(line) < 5:
            continue
        
        # Preserve reasonable indentation for paragraphs
        indent = len(original_line) - len(original_line.lstrip())
        if indent > 0 and line:
            preserved_indent = ' ' * min(indent, 8)  # Max 8 spaces indentation
            cleaned_lines.append(preserved_indent + line)
        else:
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def reconstruct_paragraphs_across_pages(text: str) -> str:
    """Aggressively reconstruct paragraphs that were broken across page boundaries"""
    lines = text.split('\n')
    reconstructed_lines = []
    current_paragraph = ""
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Empty line indicates paragraph break
        if not line:
            if current_paragraph:
                reconstructed_lines.append(current_paragraph)
                current_paragraph = ""
            reconstructed_lines.append("")  # Preserve paragraph break
            continue
        
        # Check if this line should continue the previous paragraph
        should_continue = False
        
        if current_paragraph:
            # More aggressive paragraph continuation logic
            prev_ends_complete = current_paragraph.rstrip().endswith(('.', '!', '?', ':', ';', ')', ']', '"'))
            
            # Check if line starts lowercase (strong indication of continuation)
            line_starts_lowercase = line and line[0].islower()
            
            # Check if previous line ends mid-sentence (incomplete)
            prev_ends_incomplete = (
                current_paragraph.rstrip().endswith((',', 'and', 'or', 'but', 'the', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'by')) or
                not current_paragraph.rstrip().endswith(('.', '!', '?', ':', ';')) or
                current_paragraph.count('"') % 2 == 1 or  # Unclosed quotes
                current_paragraph.count("'") % 2 == 1    # Unclosed single quotes
            )
            
            # Look for obvious new paragraph starters
            line_starts_new_paragraph = (
                re.match(r'^(Chapter|Section|Part|\d+\.|•|\*|-)\s', line) or  # Headers, lists
                line.isupper() or  # ALL CAPS (likely headers)
                line.startswith('"') or  # Direct quotes at start
                re.match(r'^[A-Z][a-z]+,', line) or  # "Name, ..." patterns
                re.match(r'^In \d{4}', line) or  # Year references
                re.match(r'^(However|Therefore|Moreover|Furthermore|Meanwhile|Consequently),', line)  # Transition words
            )
            
            # More permissive continuation logic
            should_continue = (
                (prev_ends_incomplete or line_starts_lowercase) and  # Basic continuation indicators
                not line_starts_new_paragraph and  # But not if it's clearly a new paragraph
                not (line[0].isupper() and len(line.split()) > 8)  # Unless it's a very long capitalized sentence
            )
            
            # Special cases for dialog and quotes
            if current_paragraph.count('"') % 2 == 1:  # Unclosed quote
                should_continue = True  # Always continue until quote closes
                
            # Don't break in the middle of parenthetical statements
            if current_paragraph.count('(') > current_paragraph.count(')'):
                should_continue = True
                
            # Be more aggressive with common continuation patterns
            common_continuations = [
                r'^(and|or|but|yet|so|for|nor)\s',
                r'^(because|since|although|though|while|whereas)\s',
                r'^(that|which|who|whom|whose|where|when|why|how)\s',
                r'^(to|from|with|without|under|over|through|during)\s',
                r'^(he|she|it|they|we|you|I)\s',
            ]
            
            for pattern in common_continuations:
                if re.match(pattern, line, re.IGNORECASE):
                    should_continue = True
                    break
                    
            # Don't continue if the line is very short and capitalized (likely a header)
            if len(line.split()) <= 3 and line[0].isupper() and not line_starts_lowercase:
                should_continue = False
        
        if should_continue:
            # Join with space, ensuring no double spaces
            current_paragraph = (current_paragraph + " " + line).strip()
            # Clean up any double spaces that might have been introduced
            current_paragraph = re.sub(r'\s+', ' ', current_paragraph)
        else:
            # Start new paragraph
            if current_paragraph:
                reconstructed_lines.append(current_paragraph)
            current_paragraph = line
    
    # Add any remaining paragraph
    if current_paragraph:
        reconstructed_lines.append(current_paragraph)
    
    # Post-process to extend paragraphs further if needed
    final_lines = []
    i = 0
    while i < len(reconstructed_lines):
        line = reconstructed_lines[i]
        
        # Skip empty lines
        if not line:
            final_lines.append(line)
            i += 1
            continue
            
        # Look ahead to see if next non-empty line should be merged
        j = i + 1
        while j < len(reconstructed_lines) and not reconstructed_lines[j]:
            j += 1
            
        if j < len(reconstructed_lines):
            next_line = reconstructed_lines[j]
            # Merge if the current line seems incomplete or next line continues thought
            if (not line.rstrip().endswith(('.', '!', '?')) and 
                next_line and next_line[0].islower() and
                not next_line.startswith(('chapter', 'section', 'part'))):
                # Merge the lines
                line = line + " " + next_line
                # Skip the merged line and empty lines in between
                i = j + 1
            else:
                i += 1
        else:
            i += 1
            
        final_lines.append(line)
    
    # Join lines and clean up excessive whitespace
    result = "\n".join(final_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines
    result = re.sub(r' {2,}', ' ', result)  # Remove multiple spaces
    
    return result.strip()

def has_significant_text_content(image_bytes: bytes, min_text_length: int = 50) -> bool:
    """Check if image contains significant text using OCR"""
    try:
        # Convert bytes to PIL Image for OCR
        import io
        from PIL import Image
        
        img = Image.open(io.BytesIO(image_bytes))
        
        # Resize large images for faster OCR
        if img.width > 1000 or img.height > 1000:
            img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(img, config='--psm 6')
        
        # Remove whitespace and check if substantial text exists
        clean_text = ''.join(text.split())
        
        # Consider it text-heavy if it has significant readable content
        return len(clean_text) >= min_text_length
        
    except Exception as e:
        # If OCR fails, be conservative and assume it might be text
        return True

def extract_images(pdf_path: Path, output_dir: Path, min_width=100, min_height=100, max_ratio=25) -> int:
    """Extract actual images from PDF with more permissive filtering"""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    img_count = 0
    page_dimensions = []
    
    # Get typical page dimensions for filtering
    for page_num in range(min(5, len(doc))):
        page = doc.load_page(page_num)
        rect = page.rect
        page_dimensions.append((rect.width, rect.height))
    
    if page_dimensions:
        avg_page_width = sum(w for w, h in page_dimensions) / len(page_dimensions)
        avg_page_height = sum(h for w, h in page_dimensions) / len(page_dimensions)
    else:
        avg_page_width = avg_page_height = 600  # fallback
    
    print(f"Average page dimensions: {avg_page_width:.0f}x{avg_page_height:.0f}")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        
        print(f"Page {page_num+1}: Found {len(image_list)} potential images")
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            
            print(f"  Image {img_index+1}: {width}x{height}px")
            
            # Skip tiny images (reduced threshold)
            if width < min_width or height < min_height:
                print(f"    Skipped: Too small ({width}x{height})")
                continue
            
            # More lenient page scan filtering - only skip if BOTH dimensions are very large
            width_ratio = width / avg_page_width if avg_page_width > 0 else 0
            height_ratio = height / avg_page_height if avg_page_height > 0 else 0
            
            # Only skip if both dimensions are more than 85% of page size (full page scans)
            if width_ratio > 0.85 and height_ratio > 0.85:
                print(f"    Skipped: Likely full page scan ({width_ratio:.1%} x {height_ratio:.1%} of page)")
                continue
            
            # Skip images with extreme aspect ratios (artifacts)
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            if aspect_ratio > max_ratio:
                print(f"    Skipped: Extreme aspect ratio ({aspect_ratio:.1f}:1)")
                continue
            
            # Process image
            try:
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Only process supported formats
                if image_ext.lower() not in ["jpeg", "jpg", "png", "gif", "bmp"]:
                    print(f"    Skipped: Unsupported format ({image_ext})")
                    continue
                
                # Optional text filtering - only skip if image is VERY text-heavy
                try:
                    if has_significant_text_content(image_bytes, min_text_length=200):
                        print(f"    Skipped: Text-heavy image (>200 chars detected)")
                        continue
                except Exception as ocr_e:
                    print(f"    OCR failed ({ocr_e}), keeping image")
                    pass  # If OCR fails, keep the image
                
                # Save the image if it passes all filters
                img_count += 1
                img_filename = output_dir / f"fig_{page_num+1:03d}_{img_index+1:02d}.{image_ext}"
                with open(img_filename, "wb") as f:
                    f.write(image_bytes)
                print(f"    ✅ Saved: {img_filename.name} ({width}x{height}px)")
                        
            except Exception as e:
                print(f"    ❌ Error processing image {img_index}: {e}")
                continue
    
    print(f"\n📸 Total images extracted: {img_count}")
    return img_count


def pdf_to_raw(
    pdf_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[str], None]] = None,
    clean_text: bool = True,
    detect_speakers: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as doc:

        if progress_callback:
            progress_callback("Extracting metadata...")
        title, author = extract_metadata(pdf)
        if not title:
            title = "Unknown Title"
        if not author:
            author = "Unknown Author"

        if progress_callback:
            progress_callback("Extracting printed page numbers...")
        printed_page_map = extract_printed_page_numbers(pdf)

        if progress_callback:
            progress_callback("Extracting TOC from bookmarks...")
        toc_bookmarks = extract_toc_from_bookmarks(doc)

        if progress_callback:
            progress_callback("Extracting TOC candidates from text...")
        toc_candidates = extract_toc_candidates(pdf, printed_page_map)

        # Combine TOCs, normalize to 3 elements per entry
        combined_toc_dict = {}
        for t, p, *rest in toc_bookmarks:
            combined_toc_dict[(t.lower(), p)] = (t, p, None)
        for t, p, pp in toc_candidates:
            key = (t.lower(), p)
            if key not in combined_toc_dict:
                combined_toc_dict[key] = (t, p, pp)
        combined_toc = list(combined_toc_dict.values())

        if progress_callback:
            progress_callback("Detecting chapter start pages with PyMuPDF...")
        chapter_starts = find_chapter_starts_fitz(doc)

        margin_page_numbers = sorted(set(printed_page_map.values()))

        # Postprocess TOC (remove duplicates and multiple chapters on same page)
        toc = postprocess_toc(combined_toc, chapter_starts, margin_page_numbers)

        # Detect missing chapters by numbering in titles and log them
        missing_chapters = detect_missing_chapters(toc)
        if missing_chapters:
            warning_msg = f"Warning: Missing chapters detected: {missing_chapters}"
            print(warning_msg)
            if progress_callback:
                progress_callback(warning_msg)

        if not toc:
            raise RuntimeError("Failed to extract valid TOC")

        if progress_callback:
            progress_callback(f"Final TOC entries: {len(toc)}")

        # Extract full text for multilingual language detection
        full_text = ""
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"
        
        # Detect primary and secondary languages
        language, language_info = detect_multilingual_content(full_text)

        # Extract images, filtered by size, save in '04_images'
        images_dir = output_dir / "04_images"
        if progress_callback:
            progress_callback("Extracting images (filtered by size)...")
        img_count = extract_images(pdf_path, images_dir)

        # Split TOC into front matter, main body, rear matter by keywords & numbering
        front_matter_toc = []
        main_body_toc = []
        rear_matter_toc = []

        # Determine first main body chapter index (chapter 1 or first numbered chapter)
        main_body_start_idx = 0
        for idx, (toc_title, _, _) in enumerate(toc):
            if re.match(r"^(chapter\s*)?1[\.\s\-:]", toc_title.lower()) or re.match(r"^1[\s\.\-:]", toc_title.lower()):
                main_body_start_idx = idx
                break

        for idx, entry in enumerate(toc):
            title = entry[0]
            if idx < main_body_start_idx:
                front_matter_toc.append(entry)
            elif is_rear_matter_title(title):
                rear_matter_toc.append(entry)
            else:
                main_body_toc.append(entry)

        front_matter_toc.sort(key=lambda x: x[1])
        main_body_toc.sort(key=lambda x: x[1])
        rear_matter_toc.sort(key=lambda x: x[1])

        # Define page ranges for each section
        front_pages = range(0, front_matter_toc[-1][1] + 1) if front_matter_toc else range(0)
        main_pages = range(main_body_toc[0][1], main_body_toc[-1][1] + 1) if main_body_toc else range(0)
        rear_start_page = rear_matter_toc[0][1] if rear_matter_toc else (main_body_toc[-1][1] + 1 if main_body_toc else 0)
        rear_pages = range(rear_start_page, len(pdf.pages))

        def extract_pages_to_text(pages):
            text = ""
            for i in pages:
                page = pdf.pages[i]
                page_text = page.extract_text() or ""
                if is_page_text_low_quality(page_text):
                    fitz_page = doc.load_page(i)
                    page_text = ocr_page_fitz(fitz_page)
                
                # Clean the page text
                cleaned_page_text = clean_page_text(page_text, i + 1)
                if cleaned_page_text.strip():
                    text += "\n\n" + cleaned_page_text.strip()
            
            # Reconstruct paragraphs across page breaks
            return reconstruct_paragraphs_across_pages(text.strip())

        # Extract and save sections
        front_text = extract_pages_to_text(front_pages)
        if front_text:
            save_text_file(output_dir / "01_front_matter.txt", front_text)
            if progress_callback:
                progress_callback("Saved front matter.")

        # Extract individual chapters instead of one large main body
        chapters_dir = output_dir / "chapters"
        chapters_dir.mkdir(exist_ok=True)
        
        chapter_entries = []
        for i, (chapter_title, page_idx, printed_page) in enumerate(main_body_toc):
            # Determine chapter page range
            start_page = page_idx
            if i + 1 < len(main_body_toc):
                end_page = main_body_toc[i + 1][1]  # Next chapter's start page
            else:
                # Last chapter extends to end of main body or start of rear matter
                end_page = rear_matter_toc[0][1] if rear_matter_toc else len(pdf.pages)
            
            # Extract chapter text
            chapter_pages = range(start_page, end_page)
            chapter_text = extract_pages_to_text(chapter_pages)
            
            if chapter_text.strip():
                # Create safe filename from chapter title
                safe_title = re.sub(r'[^a-zA-Z0-9_\-\s]', '', chapter_title.strip())
                safe_title = re.sub(r'\s+', '_', safe_title)[:50]  # Limit length
                chapter_filename = f"chapter_{i+1:02d}_{safe_title}.txt"
                
                # Save chapter file
                chapter_path = chapters_dir / chapter_filename
                
                # Add book and chapter header with metadata
                chapter_header = f"{title} by {author}\n\n"
                chapter_header += f"Chapter {i+1}: {chapter_title}\n\n"
                
                # Add chapter ending
                chapter_ending = f"\n\nEnd of Chapter {i+1} of {title} by {author}."
                
                full_chapter_text = chapter_header + chapter_text + chapter_ending
                save_text_file(chapter_path, full_chapter_text)
                
                chapter_entries.append({
                    'number': i + 1,
                    'title': chapter_title,
                    'filename': chapter_filename,
                    'pages': f"{start_page+1}-{end_page}",
                    'printed_page': printed_page
                })
                
                if progress_callback:
                    progress_callback(f"Saved Chapter {i+1}: {chapter_title[:30]}...")
        
        # Create a chapter index file
        chapter_index_lines = ["CHAPTER INDEX\n" + "=" * 50 + "\n"]
        for entry in chapter_entries:
            chapter_index_lines.append(
                f"Chapter {entry['number']:02d}: {entry['title']}\n"
                f"  File: {entry['filename']}\n"
                f"  Pages: {entry['pages']}\n"
                f"  Printed Page: {entry['printed_page'] or 'N/A'}\n"
            )
        
        save_text_file(output_dir / "02_chapter_index.txt", "\n".join(chapter_index_lines))
        
        # Also save the full main body for backward compatibility
        main_text = extract_pages_to_text(main_pages)
        if main_text:
            save_text_file(output_dir / "02_main_body.txt", main_text)
            if progress_callback:
                progress_callback(f"Created {len(chapter_entries)} individual chapter files.")

        rear_text = extract_pages_to_text(rear_pages)
        if rear_text:
            save_text_file(output_dir / "03_rear_matter.txt", rear_text)
            if progress_callback:
                progress_callback("Saved rear matter.")

        # Save TOC preview with metadata header, language, and image count
        language_details = ", ".join([f"{lang}: {pct}%" for lang, pct in language_info.items()])
        toc_lines = [
            f"Title: {title}",
            f"Author: {author}",
            f"Primary Language: {language}",
            f"Language breakdown: {language_details}",
            f"Images extracted: {img_count}",
            "Table of Contents:",
        ]
        toc_lines += format_toc_entries(front_matter_toc, 1)
        toc_lines += format_toc_entries(main_body_toc, 2)
        toc_lines += format_toc_entries(rear_matter_toc, 3)
        save_text_file(output_dir / "00_contents.txt", "\n".join(toc_lines))

        if progress_callback:
            progress_callback("Extraction complete.")

        return toc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract raw text and TOC from PDF with PyMuPDF and OCR fallback.")
    parser.add_argument("pdf_path", type=Path, help="Input PDF file path")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    args = parser.parse_args()

    def print_progress(msg):
        print(msg)

    pdf_to_raw(args.pdf_path, args.output_dir, print_progress)
