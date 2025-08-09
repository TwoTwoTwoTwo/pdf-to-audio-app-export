#!/usr/bin/env python3
"""
Speaker Detection Module
Detects multiple speakers in text and assigns different voices
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import json
from pathlib import Path

class Speaker:
    def __init__(self, name: str, voice_id: str = None, characteristics: Dict = None):
        self.name = name
        self.voice_id = voice_id or f"speaker_{name.lower().replace(' ', '_')}"
        self.characteristics = characteristics or {}
        self.dialogue_count = 0
        self.word_count = 0
        
    def to_dict(self):
        return {
            "name": self.name,
            "voice_id": self.voice_id,
            "characteristics": self.characteristics,
            "dialogue_count": self.dialogue_count,
            "word_count": self.word_count
        }

class SpeakerDetector:
    def __init__(self):
        self.speakers = {}
        self.narrator_voice = "narrator"
        
    def detect_speakers_in_text(self, text: str) -> Dict[str, Speaker]:
        """Detect speakers in text based on dialogue patterns"""
        
        # Always create a narrator speaker for the entire text
        narrator = Speaker("Narrator")
        narrator.word_count = len(text.split())
        narrator.dialogue_count = 1  # The entire text is considered one "dialogue" by the narrator
        narrator.characteristics = {
            "formality_score": 0.5,
            "excitement_level": 0.3,
            "calmness_level": 0.7,
            "avg_sentence_length": 15.0,
            "total_words": narrator.word_count,
            "dialogue_snippets": [text[:200] + "..." if len(text) > 200 else text]
        }
        
        # Start with narrator as the default
        speakers = {"Narrator": narrator}
        
        # Only look for additional speakers if there are clear dialogue patterns
        dialogue_patterns = [
            # Direct speech with attribution
            r'"([^"]+)"\s*,?\s*(?:said|asked|replied|answered|whispered|shouted|exclaimed|declared|announced|stated|remarked|observed|noted|added|continued|explained|insisted|protested|admitted|confessed|agreed|disagreed|argued|countered|interrupted|interjected)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|asked|replied|answered|whispered|shouted|exclaimed|declared|announced|stated|remarked|observed|noted|added|continued|explained|insisted|protested|admitted|confessed|agreed|disagreed|argued|countered|interrupted|interjected)\s*,?\s*"([^"]+)"',
            
            # Dialogue with colons (script/interview format)
            r'^([A-Z][A-Z\s]+):\s*(.+)$',  # ALL CAPS speaker names
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):\s*(.+)$',  # Title Case speaker names
            
            # Narrative descriptions of speech
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:told|informed|warned|advised|suggested|proposed|mentioned|indicated|revealed|disclosed|confided|shared|expressed|conveyed|communicated|reported|recounted|described|narrated)\s+(?:me|us|them|him|her|everyone|the\s+\w+)',
        ]
        
        speaker_dialogue = defaultdict(list)
        speaker_word_counts = defaultdict(int)
        
        lines = text.splitlines()
        dialogue_found = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try each dialogue pattern
            for pattern in dialogue_patterns:
                matches = re.findall(pattern, line, re.MULTILINE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        # Extract speaker and dialogue
                        if pattern.endswith('([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)'):
                            # Speaker name is second group
                            dialogue_text, speaker_name = match
                        else:
                            # Speaker name is first group
                            speaker_name, dialogue_text = match
                        
                        speaker_name = speaker_name.strip()
                        dialogue_text = dialogue_text.strip()
                        
                        # Filter out common false positives
                        if self._is_valid_speaker_name(speaker_name):
                            speaker_dialogue[speaker_name].append(dialogue_text)
                            speaker_word_counts[speaker_name] += len(dialogue_text.split())
                            dialogue_found = True
        
        # Only add additional speakers if we found significant dialogue patterns
        if dialogue_found:
            for speaker_name, dialogues in speaker_dialogue.items():
                if len(dialogues) >= 2:  # Only include speakers with at least 2 dialogue instances
                    speaker = Speaker(speaker_name)
                    speaker.dialogue_count = len(dialogues)
                    speaker.word_count = speaker_word_counts[speaker_name]
                    speaker.characteristics = self._analyze_speaker_characteristics(dialogues)
                    speakers[speaker_name] = speaker
        
        return speakers
    
    def _is_valid_speaker_name(self, name: str) -> bool:
        """Check if a name is likely a valid speaker name"""
        
        # Filter out common false positives
        false_positives = {
            'said', 'asked', 'replied', 'answered', 'told', 'explained', 'continued',
            'however', 'therefore', 'moreover', 'furthermore', 'meanwhile', 'consequently',
            'chapter', 'section', 'part', 'page', 'book', 'story', 'novel', 'author',
            'narrator', 'voice', 'character', 'person', 'people', 'everyone', 'someone',
            'anyone', 'nobody', 'everybody', 'somebody', 'anybody'
        }
        
        name_lower = name.lower()
        
        # Must be reasonable length
        if len(name) < 2 or len(name) > 50:
            return False
            
        # Must not be a common false positive
        if name_lower in false_positives:
            return False
            
        # Must contain at least one letter
        if not any(c.isalpha() for c in name):
            return False
            
        # Must not be mostly numbers
        if sum(1 for c in name if c.isdigit()) > len(name) * 0.5:
            return False
            
        # Should look like a proper name (title case or all caps)
        words = name.split()
        if len(words) <= 3:  # Reasonable number of name parts
            return True
            
        return False
    
    def _analyze_speaker_characteristics(self, dialogues: List[str]) -> Dict:
        """Analyze speaker characteristics from their dialogue"""
        
        all_text = " ".join(dialogues)
        
        # Analyze formality
        formal_indicators = len(re.findall(r'\b(therefore|however|furthermore|moreover|consequently|nevertheless|nonetheless|accordingly|thus|hence)\b', all_text, re.I))
        casual_indicators = len(re.findall(r'\b(yeah|okay|yep|nope|gonna|wanna|gotta|dunno|kinda|sorta)\b', all_text, re.I))
        
        # Analyze emotion/tone
        excited_indicators = len(re.findall(r'[!]+|[?!]+|\b(amazing|awesome|fantastic|incredible|wonderful|excellent)\b', all_text, re.I))
        calm_indicators = len(re.findall(r'\b(certainly|indeed|perhaps|possibly|naturally|obviously|clearly)\b', all_text, re.I))
        
        # Analyze sentence length (complexity)
        sentences = re.split(r'[.!?]+', all_text)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
        
        return {
            "formality_score": formal_indicators / max(formal_indicators + casual_indicators, 1),
            "excitement_level": excited_indicators / max(len(sentences), 1),
            "calmness_level": calm_indicators / max(len(sentences), 1),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "total_words": len(all_text.split()),
            "dialogue_snippets": dialogues[:3]  # Sample dialogues
        }
    
    def assign_voices_to_speakers(self, speakers: Dict[str, Speaker]) -> Dict[str, str]:
        """Assign appropriate TTS voices to detected speakers"""
        
        # Available TTS voice models with characteristics
        available_voices = {
            "jenny_female_us": {
                "gender": "female",
                "accent": "US",
                "characteristics": ["warm", "professional", "clear"],
                "model": "tts_models/en/ljspeech/tacotron2-DDC"
            },
            "david_male_us": {
                "gender": "male", 
                "accent": "US",
                "characteristics": ["authoritative", "deep", "formal"],
                "model": "tts_models/en/vctk/vits"
            },
            "mary_female_uk": {
                "gender": "female",
                "accent": "UK", 
                "characteristics": ["gentle", "sophisticated", "calm"],
                "model": "tts_models/en/vctk/vits"
            },
            "narrator_neutral": {
                "gender": "neutral",
                "accent": "US",
                "characteristics": ["steady", "clear", "narrative"],
                "model": "tts_models/en/ljspeech/glow-tts"
            }
        }
        
        voice_assignments = {}
        used_voices = set()
        
        # Sort speakers by dialogue count (most active first)
        sorted_speakers = sorted(speakers.items(), key=lambda x: x[1].dialogue_count, reverse=True)
        
        for i, (speaker_name, speaker) in enumerate(sorted_speakers):
            # Simple voice assignment based on characteristics and availability
            best_voice = None
            
            # Try to match characteristics
            for voice_id, voice_info in available_voices.items():
                if voice_id in used_voices:
                    continue
                    
                # Simple assignment based on formality and activity
                if speaker.characteristics.get("formality_score", 0) > 0.7:
                    if "formal" in voice_info["characteristics"] or "authoritative" in voice_info["characteristics"]:
                        best_voice = voice_id
                        break
                elif speaker.characteristics.get("excitement_level", 0) > 0.5:
                    if "warm" in voice_info["characteristics"]:
                        best_voice = voice_id
                        break
                elif speaker.characteristics.get("calmness_level", 0) > 0.5:
                    if "calm" in voice_info["characteristics"] or "gentle" in voice_info["characteristics"]:
                        best_voice = voice_id
                        break
            
            # Fallback: assign any available voice
            if not best_voice:
                available = [v for v in available_voices.keys() if v not in used_voices]
                if available:
                    best_voice = available[0]
            
            if best_voice:
                voice_assignments[speaker_name] = {
                    "voice_id": best_voice,
                    "model": available_voices[best_voice]["model"],
                    "characteristics": available_voices[best_voice]["characteristics"]
                }
                used_voices.add(best_voice)
                speaker.voice_id = best_voice
        
        return voice_assignments
    
    def create_speaker_manifest(self, text: str, output_path: Path) -> Dict:
        """Create a complete speaker manifest for the text"""
        
        # Detect speakers
        speakers = self.detect_speakers_in_text(text)
        
        # Assign voices
        voice_assignments = self.assign_voices_to_speakers(speakers)
        
        # Create manifest
        manifest = {
            "total_speakers_detected": len(speakers),
            "narrator_voice": "narrator_neutral",
            "speakers": {}
        }
        
        for speaker_name, speaker in speakers.items():
            manifest["speakers"][speaker_name] = {
                **speaker.to_dict(),
                "voice_assignment": voice_assignments.get(speaker_name, {
                    "voice_id": "narrator_neutral",
                    "model": "tts_models/en/ljspeech/glow-tts",
                    "characteristics": ["neutral"]
                })
            }
        
        # Save manifest
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return manifest
    
    def split_text_by_speaker(self, text: str) -> List[Dict[str, str]]:
        """Split text into segments with speaker assignments"""
        
        segments = []
        current_speaker = "narrator"
        current_text = ""
        
        # Dialogue patterns for splitting
        dialogue_patterns = [
            r'"([^"]+)"\s*,?\s*(?:said|asked|replied)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|asked|replied)\s*,?\s*"([^"]+)"',
            r'^([A-Z][A-Z\s]+):\s*(.+)$',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):\s*(.+)$',
        ]
        
        lines = text.splitlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_text:
                    current_text += "\n"
                continue
            
            # Check if line contains dialogue
            found_dialogue = False
            for pattern in dialogue_patterns:
                match = re.search(pattern, line)
                if match:
                    found_dialogue = True
                    
                    # Add current text segment if it exists
                    if current_text.strip():
                        segments.append({
                            "speaker": current_speaker,
                            "text": current_text.strip()
                        })
                        current_text = ""
                    
                    # Extract speaker and dialogue
                    if pattern.endswith('([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)'):
                        dialogue_text, speaker_name = match.groups()
                    else:
                        speaker_name, dialogue_text = match.groups()
                    
                    speaker_name = speaker_name.strip()
                    
                    # Add dialogue segment
                    if self._is_valid_speaker_name(speaker_name):
                        segments.append({
                            "speaker": speaker_name,
                            "text": dialogue_text.strip()
                        })
                        current_speaker = "narrator"  # Reset to narrator after dialogue
                    else:
                        current_text += line + "\n"
                    
                    break
            
            if not found_dialogue:
                current_text += line + "\n"
        
        # Add any remaining text
        if current_text.strip():
            segments.append({
                "speaker": current_speaker,
                "text": current_text.strip()
            })
        
        return segments

def detect_speakers_in_file(file_path: Path) -> Dict:
    """Convenience function to detect speakers in a single file"""
    
    detector = SpeakerDetector()
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create output path for manifest
    manifest_path = file_path.parent / f"{file_path.stem}_speakers.json"
    
    # Detect and create manifest
    manifest = detector.create_speaker_manifest(text, manifest_path)
    
    return manifest

def main():
    """Command line interface for speaker detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect speakers in text files for multi-voice TTS")
    parser.add_argument("input_path", type=Path, help="Input text file or directory")
    parser.add_argument("--output", "-o", type=Path, help="Output directory for speaker manifests")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = args.output or input_path.parent
    
    if input_path.is_file():
        # Process single file
        print(f"🔍 Detecting speakers in {input_path.name}...")
        manifest = detect_speakers_in_file(input_path)
        print(f"✅ Found {manifest['total_speakers_detected']} speakers")
        for speaker_name, speaker_info in manifest['speakers'].items():
            print(f"  • {speaker_name}: {speaker_info['dialogue_count']} dialogues, {speaker_info['word_count']} words")
    
    elif input_path.is_dir():
        # Process directory
        text_files = list(input_path.glob("*.txt"))
        print(f"🔍 Processing {len(text_files)} text files...")
        
        total_speakers = set()
        for text_file in text_files:
            print(f"  Processing {text_file.name}...")
            manifest = detect_speakers_in_file(text_file)
            total_speakers.update(manifest['speakers'].keys())
        
        print(f"✅ Found {len(total_speakers)} unique speakers across all files")
        for speaker in sorted(total_speakers):
            print(f"  • {speaker}")
    
    else:
        print(f"❌ Input path {input_path} not found")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
