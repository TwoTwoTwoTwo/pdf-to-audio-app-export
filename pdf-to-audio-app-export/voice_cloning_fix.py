"""
Voice Cloning Detection and Mapping Fix

This module provides proper mapping between voice names in the app
and the actual cloned voice data stored in session state.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st

def extract_clean_voice_name(voice_display_name: str) -> str:
    """
    Extract clean voice name from display format.
    
    Examples:
    "🎬 Morgan Freeman (Cloned)" -> "Morgan Freeman"
    "🎬 Custom Voice (Cloned)" -> "Custom Voice"
    "cloned_morgan_freeman" -> "Morgan Freeman"
    """
    # Remove emoji and (Cloned) suffix
    clean_name = voice_display_name.replace("🎬 ", "").replace(" (Cloned)", "").strip()
    
    # Handle cloned_ prefix format
    if clean_name.startswith("cloned_"):
        clean_name = clean_name.replace("cloned_", "").replace("_", " ").title()
    
    return clean_name

def find_cloned_voice_data(voice_name: str, cloned_voices_data: Optional[Dict] = None) -> Optional[str]:
    """
    Find the speaker_wav file for a cloned voice by matching names.
    
    Args:
        voice_name: The voice name to search for
        cloned_voices_data: Optional cloned voices data dict
    
    Returns:
        Path to speaker_wav file if found, None otherwise
    """
    print(f"🔍 Searching for cloned voice: '{voice_name}'")
    
    # Clean the input voice name
    clean_voice_name = extract_clean_voice_name(voice_name)
    print(f"   Clean name: '{clean_voice_name}'")
    
    # Try to get from provided data first
    if cloned_voices_data:
        print(f"   Checking provided cloned_voices_data with {len(cloned_voices_data)} voices")
        
        for cloned_name, cloned_data in cloned_voices_data.items():
            clean_cloned_name = extract_clean_voice_name(cloned_name)
            print(f"   Comparing '{clean_voice_name}' with '{clean_cloned_name}'")
            
            # Try exact match first
            if clean_voice_name.lower() == clean_cloned_name.lower():
                speaker_wav = cloned_data.get('speaker_wav')
                if speaker_wav and os.path.exists(speaker_wav):
                    print(f"   ✅ Found exact match: {speaker_wav}")
                    return speaker_wav
            
            # Try partial match
            elif (clean_voice_name.lower() in clean_cloned_name.lower() or 
                  clean_cloned_name.lower() in clean_voice_name.lower()):
                speaker_wav = cloned_data.get('speaker_wav')
                if speaker_wav and os.path.exists(speaker_wav):
                    print(f"   ✅ Found partial match: {speaker_wav}")
                    return speaker_wav
    
    # Try session state as fallback
    try:
        if hasattr(st, 'session_state') and 'cloned_voices' in st.session_state:
            session_voices = st.session_state['cloned_voices']
            print(f"   Checking session state with {len(session_voices)} voices")
            
            for session_name, session_data in session_voices.items():
                clean_session_name = extract_clean_voice_name(session_name)
                print(f"   Comparing '{clean_voice_name}' with session '{clean_session_name}'")
                
                # Try exact match
                if clean_voice_name.lower() == clean_session_name.lower():
                    speaker_wav = session_data.get('speaker_wav')
                    if speaker_wav and os.path.exists(speaker_wav):
                        print(f"   ✅ Found session exact match: {speaker_wav}")
                        return speaker_wav
                
                # Try partial match
                elif (clean_voice_name.lower() in clean_session_name.lower() or 
                      clean_session_name.lower() in clean_voice_name.lower()):
                    speaker_wav = session_data.get('speaker_wav')
                    if speaker_wav and os.path.exists(speaker_wav):
                        print(f"   ✅ Found session partial match: {speaker_wav}")
                        return speaker_wav
                        
    except Exception as e:
        print(f"   ⚠️ Session state access failed: {e}")
    
    print(f"   ❌ No cloned voice found for '{voice_name}'")
    return None

def is_cloned_voice(voice_name: str) -> bool:
    """
    Check if a voice name indicates a cloned voice.
    
    Args:
        voice_name: The voice name to check
    
    Returns:
        True if this appears to be a cloned voice
    """
    cloned_indicators = [
        "(Cloned)",
        "cloned_",
        "🎬",
        "(Custom)"
    ]
    
    return any(indicator in voice_name for indicator in cloned_indicators)

def debug_voice_mapping(voice_name: str, cloned_voices_data: Optional[Dict] = None):
    """
    Debug function to show voice mapping details.
    """
    print(f"\n🐛 DEBUG: Voice mapping for '{voice_name}'")
    print(f"   Is cloned voice: {is_cloned_voice(voice_name)}")
    print(f"   Clean name: '{extract_clean_voice_name(voice_name)}'")
    
    if cloned_voices_data:
        print(f"   Available cloned voices in data:")
        for name, data in cloned_voices_data.items():
            speaker_wav = data.get('speaker_wav', 'N/A')
            exists = os.path.exists(speaker_wav) if speaker_wav != 'N/A' else False
            print(f"     - '{name}' -> {speaker_wav} (exists: {exists})")
    
    try:
        if hasattr(st, 'session_state') and 'cloned_voices' in st.session_state:
            print(f"   Available cloned voices in session:")
            for name, data in st.session_state['cloned_voices'].items():
                speaker_wav = data.get('speaker_wav', 'N/A')
                exists = os.path.exists(speaker_wav) if speaker_wav != 'N/A' else False
                print(f"     - '{name}' -> {speaker_wav} (exists: {exists})")
    except Exception as e:
        print(f"   Session state not available: {e}")
    
    # Try to find the voice
    found_wav = find_cloned_voice_data(voice_name, cloned_voices_data)
    print(f"   Result: {found_wav}")
    print("🐛 END DEBUG\n")

def get_voice_for_synthesis(model_name: str, cloned_voices_data: Optional[Dict] = None) -> Optional[str]:
    """
    Get the speaker_wav file for synthesis if this is a cloned voice.
    
    Args:
        model_name: The model/voice name selected by user
        cloned_voices_data: Optional cloned voices data
    
    Returns:
        Path to speaker_wav file if this is a cloned voice, None otherwise
    """
    if not is_cloned_voice(model_name):
        return None
    
    return find_cloned_voice_data(model_name, cloned_voices_data)
