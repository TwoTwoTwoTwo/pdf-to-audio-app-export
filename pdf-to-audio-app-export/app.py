"""
PDF to Audiobook Converter - Streamlined 3-Tab Version
Clean workflow: Extract → Process → Create Audio/Video
"""

# Install XTTS import hook BEFORE any TTS/XTTS imports so GPT2InferenceModel is patched app-wide
try:
    import import_hook_xtts_patch  # auto-installs on import
    print("🔧 XTTS import hook ensured at app startup")
except Exception:
    pass

# Import the preview fix to ensure it's available
try:
    import xtts_preview_fix
    print("🎧 XTTS preview fix loaded")
except Exception:
    pass

# CRITICAL: Apply the FINAL WORKING XTTS solution FIRST
# This ensures XTTS-v2 loads properly with modern transformers
try:
    from xtts_final_working import get_final_working_tts, is_tts_ready
    print("✅ Final working XTTS implementation loaded successfully")
except ImportError:
    print("❌ Final working XTTS implementation not found - voice cloning will not work")
    get_final_working_tts = None
    is_tts_ready = lambda: False

import streamlit as st
import tempfile
import zipfile
import re
import json
from pathlib import Path
import uuid

from pdf_to_raw import pdf_to_raw
from txt_to_mp3 import convert_txt_to_mp3

try:
    from voice_editor import voice_editor_ui
except ImportError:
    voice_editor_ui = None
    
try:
    from speaker_detection import SpeakerDetector
except ImportError:
    SpeakerDetector = None
    
try:
    from video_export import VideoExporter
except ImportError:
    VideoExporter = None

st.set_page_config(page_title="PDF to Audio", layout="centered")
st.title("📘 PDF to Audiobook Converter")
st.markdown("Transform PDFs into professional audiobooks with speaker detection, voice cloning, and video export.")

# Create 2 streamlined tabs
tabs = st.tabs([
    "📄 PDF Processing & Speaker Detection",
    "🎵 Audio & Video Creation with Voice Editor"
])

temp_dir = Path(tempfile.mkdtemp())

# Initialize session state for data sharing between tabs
if 'processed_chapters' not in st.session_state:
    st.session_state.processed_chapters = []
if 'detected_speakers' not in st.session_state:
    st.session_state.detected_speakers = {}
if 'extracted_images' not in st.session_state:
    st.session_state.extracted_images = []
if 'book_metadata' not in st.session_state:
    st.session_state.book_metadata = {'title': 'My Audiobook', 'author': 'Unknown Author'}

##############################
# Tab 1: PDF Processing & Speaker Detection  
##############################
with tabs[0]:
    st.header("📄 PDF Processing & Speaker Detection")
    st.write("Upload your PDF to extract text, images, and automatically detect speakers in one step.")
    
    # PDF Upload
    uploaded_pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])
    
    if uploaded_pdf:
        pdf_path = temp_dir / uploaded_pdf.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        col1, col2 = st.columns(2)
        with col1:
            extract_images = st.checkbox("🖼️ Extract Images", value=True, help="Extract images from PDF for video creation")
        with col2:
            detect_speakers = st.checkbox("👥 Detect Speakers", value=True, help="Analyze text for multiple speakers")
        
        if st.button("🚀 Process PDF", type="primary"):
            raw_output_dir = temp_dir / "raw"
            raw_output_dir.mkdir(exist_ok=True)
            
            # Progress tracking
            progress_text = st.empty()
            progress_bar = st.progress(0)
            progress_val = [0]
            
            def update_progress(msg):
                progress_val[0] = min(progress_val[0] + 10, 90)
                progress_text.text(msg)
                progress_bar.progress(progress_val[0])
            
            try:
                # Step 1: Extract PDF content
                with st.spinner("📖 Extracting PDF content..."):
                    toc = pdf_to_raw(pdf_path, raw_output_dir, progress_callback=update_progress)
                    progress_val[0] = 60
                    progress_bar.progress(progress_val[0])
                
                # Get book metadata
                contents_file = raw_output_dir / "00_contents.txt"
                if contents_file.exists():
                    contents_text = contents_file.read_text(encoding='utf-8')
                    lines = contents_text.splitlines()
                    for line in lines:
                        if line.startswith("Title: "):
                            st.session_state.book_metadata['title'] = line[7:].strip()
                        elif line.startswith("Author: "):
                            st.session_state.book_metadata['author'] = line[8:].strip()
                
                # Step 2: Collect processed chapters
                chapters_dir = raw_output_dir / "chapters"
                st.session_state.processed_chapters = []
                if chapters_dir.exists():
                    chapter_files = sorted(list(chapters_dir.glob("*.txt")))
                    for chapter_file in chapter_files:
                        st.session_state.processed_chapters.append({
                            'name': chapter_file.name,
                            'path': chapter_file,
                            'content': chapter_file.read_text(encoding='utf-8')
                        })
                
                # Step 3: Collect extracted images  
                images_dir = raw_output_dir / "04_images"
                st.session_state.extracted_images = []
                if extract_images and images_dir.exists():
                    image_files = list(images_dir.glob("*"))
                    for img_file in image_files:
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                            st.session_state.extracted_images.append(img_file)
                
                # Step 4: Speaker Detection
                st.session_state.detected_speakers = {}
                if detect_speakers and SpeakerDetector and st.session_state.processed_chapters:
                    update_progress("🎭 Detecting speakers in text...")
                    detector = SpeakerDetector()
                    all_speakers = {}
                    
                    # Analyze each chapter for speakers
                    for chapter in st.session_state.processed_chapters:
                        speakers = detector.detect_speakers_in_text(chapter['content'])
                        for name, speaker in speakers.items():
                            if name in all_speakers:
                                all_speakers[name].dialogue_count += speaker.dialogue_count
                                all_speakers[name].word_count += speaker.word_count
                                # Merge dialogue snippets
                                existing_snippets = all_speakers[name].characteristics.get('dialogue_snippets', [])
                                new_snippets = speaker.characteristics.get('dialogue_snippets', [])
                                all_speakers[name].characteristics['dialogue_snippets'] = existing_snippets + new_snippets
                            else:
                                all_speakers[name] = speaker
                    
                    st.session_state.detected_speakers = all_speakers
                
                progress_bar.progress(100)
                progress_text.text("✅ Processing complete!")
                st.success("🎉 PDF processing complete!")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
                st.stop()
    
    # Show Results
    if st.session_state.processed_chapters:
        st.subheader("📊 Processing Results")
        
        # Book info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Chapters", len(st.session_state.processed_chapters))
        with col2:
            st.metric("🖼️ Images", len(st.session_state.extracted_images))  
        with col3:
            st.metric("🎭 Speakers", len(st.session_state.detected_speakers))
        
        # Show book metadata
        st.info(f"**📖 {st.session_state.book_metadata['title']}** by {st.session_state.book_metadata['author']}")
        
        # Chapter preview
        with st.expander("📚 View Chapters"):
            for chapter in st.session_state.processed_chapters:
                st.write(f"• **{chapter['name']}** ({len(chapter['content'])} characters)")
        
        # Speaker detection results
        if st.session_state.detected_speakers:
            with st.expander("🎭 Detected Speakers"):
                for name, speaker in sorted(st.session_state.detected_speakers.items(), 
                                          key=lambda x: x[1].dialogue_count, reverse=True):
                    st.write(f"**🎤 {name}** - {speaker.dialogue_count} dialogues, {speaker.word_count} words")
                    if hasattr(speaker, 'characteristics'):
                        formality = speaker.characteristics.get('formality_score', 0)
                        avg_length = speaker.characteristics.get('avg_sentence_length', 0)
                        st.write(f"   Formality: {formality:.2f}, Avg sentence: {avg_length} words")
        
        # Download processed files
        if st.session_state.processed_chapters:
            # Create download ZIP
            zip_path = temp_dir / "processed_content.zip"
            with zipfile.ZipFile(zip_path, "w") as zipf:
                # Add chapter files
                for chapter in st.session_state.processed_chapters:
                    zipf.write(chapter['path'], f"chapters/{chapter['name']}")
                
                # Add images
                for img_path in st.session_state.extracted_images:
                    zipf.write(img_path, f"images/{img_path.name}")
                
                # Add speaker manifest if speakers detected
                if st.session_state.detected_speakers:
                    speaker_manifest = {
                        'book_title': st.session_state.book_metadata['title'],
                        'author': st.session_state.book_metadata['author'],
                        'speakers': {}
                    }
                    for name, speaker in st.session_state.detected_speakers.items():
                        speaker_manifest['speakers'][name] = {
                            'dialogue_count': speaker.dialogue_count,
                            'word_count': speaker.word_count,
                            'characteristics': getattr(speaker, 'characteristics', {})
                        }
                    
                    manifest_content = json.dumps(speaker_manifest, indent=2)
                    zipf.writestr("speaker_manifest.json", manifest_content)
            
            with open(zip_path, "rb") as f:
                st.download_button(
                    "📦 Download Processed Content",
                    data=f,
                    file_name="processed_audiobook_content.zip",
                    mime="application/zip"
                )

##############################
# Tab 2: Audio & Video Creation
##############################
with tabs[1]:
    st.header("🎵 Audio & Video Creation")
    st.write("Clean workflow: Create Voices → Assign Voices → Generate Audiobook")
    
    # Check if we have processed content
    if not st.session_state.processed_chapters:
        st.warning("⚠️ Please process a PDF in Tab 1 first, or upload chapter files below.")
        
        # Alternative: Upload chapter files directly
        st.subheader("📂 Or Upload Chapter Files Directly")
        uploaded_files = st.file_uploader(
            "Upload Chapter Text Files",
            type=["txt"],
            accept_multiple_files=True,
            help="Upload .txt files if you've already processed them elsewhere"
        )
        
        if uploaded_files:
            st.session_state.processed_chapters = []
            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode('utf-8')
                st.session_state.processed_chapters.append({
                    'name': uploaded_file.name,
                    'path': None,
                    'content': content
                })
            st.success(f"✅ Loaded {len(uploaded_files)} chapter files")
    
    if st.session_state.processed_chapters:
        st.success(f"📚 Ready to process {len(st.session_state.processed_chapters)} chapters")
        
        # Initialize available voices in session state with high-quality models including XTTS-v2 for voice cloning
        if 'available_voices' not in st.session_state:
            st.session_state.available_voices = [
                "tts_models/en/vctk/vits",  # Multi-speaker VCTK (highest quality, most reliable)
                "tts_models/multilingual/multi-dataset/xtts_v2",  # XTTS-v2 for voice cloning (ESSENTIAL)
                "tts_models/en/ljspeech/glow-tts",  # GlowTTS (better than Tacotron)
                "tts_models/en/ljspeech/fast_pitch",  # FastPitch (modern architecture)
            ]
        
        # STEP 1: Voice Creation & Management
        st.subheader("🎤 Step 1: Create & Manage Voices")
        
        # Voice Cloning Options
        voice_cloning_tabs = st.tabs(["🎬 Clone from YouTube", "🎵 Clone from MP3 Upload"])
        
        # YouTube Voice Cloning Tab
        with voice_cloning_tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                youtube_url = st.text_input(
                    "📺 YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    help="URL of video with clear speech",
                    key="youtube_clone_url"
                )
            
            with col2:
                speaker_name = st.text_input(
                    "🎤 Voice Name", 
                    placeholder="e.g., Morgan Freeman",
                    help="Name for this voice",
                    key="youtube_clone_name"
                )
            
            if youtube_url and speaker_name:
                st.info("⚠️ Only use content you have permission to use.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Clone Voice", key="clone_voice_btn"):
                        try:
                            from voice_editor import clone_voice_from_youtube_integrated
                            
                            with st.spinner(f"Cloning voice for {speaker_name}..."):
                                success = clone_voice_from_youtube_integrated(youtube_url, speaker_name)
                            
                            if success:
                                st.success(f"✅ Voice '{speaker_name}' cloned successfully!")
                                # Add to available voices
                                voice_id = f"cloned_{speaker_name.lower().replace(' ', '_')}"
                                if voice_id not in st.session_state.available_voices:
                                    st.session_state.available_voices.append(voice_id)
                                st.rerun()
                            else:
                                st.error("❌ Voice cloning failed.")
                            
                        except ImportError:
                            st.error("❌ Install required packages: `pip install yt-dlp pydub`")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                
                with col2:
                    if st.button("🎵 Test Voice", key="test_cloned_voice"):
                        if youtube_url and speaker_name:
                            # Test the voice that would be cloned
                            test_text = "Hello! This is a preview of how the cloned voice will sound. The system will extract the speaker's voice characteristics from the YouTube video to create a custom voice model."
                            
                            # Check if this voice is already cloned
                            cloned_voices = st.session_state.get('cloned_voices', {})
                            if speaker_name in cloned_voices:
                                st.info(f"🎬 Voice '{speaker_name}' already exists! Testing cloned voice...")
                                
                                # Test the existing cloned voice using XTTS-v2
                                try:
                                    from voice_editor import generate_voice_preview
                                    
                                    # Create voice config for the cloned voice - FORCE XTTS-v2
                                    voice_config = {
                                        speaker_name: {
                                            "model": "tts_models/multilingual/multi-dataset/xtts_v2",  # XTTS-v2 REQUIRED
                                            "type": "cloned",
                                            "speaker_wav": cloned_voices[speaker_name]['speaker_wav'],
                                            "description": f"🎆 XTTS-v2 cloned voice from YouTube"
                                        }
                                    }
                                    
                                    st.info("🎬 Testing with XTTS-v2 voice cloning model...")
                                    generate_voice_preview(test_text, speaker_name, voice_config)
                                
                                except Exception as e:
                                    st.error(f"❌ Failed to test cloned voice: {e}")
                                    st.info("💡 Try cloning the voice again if the test fails.")
                            else:
                                st.warning("⚠️ Voice not cloned yet. Clone the voice first, then test it.")
                        else:
                            st.warning("⚠️ Please provide both YouTube URL and voice name first.")
        
        # MP3 Upload Voice Cloning Tab  
        with voice_cloning_tabs[1]:
            st.write("Upload an MP3 file to clone the voice from that audio.")
            
            col1, col2 = st.columns(2)
            with col1:
                uploaded_voice_file = st.file_uploader(
                    "🎵 Upload Voice MP3 File",
                    type=["mp3", "wav", "m4a", "flac", "ogg"],
                    help="Upload audio file with clear speech (3-30 seconds works best)",
                    key="voice_file_upload"
                )
            
            with col2:
                mp3_speaker_name = st.text_input(
                    "🎤 Voice Name", 
                    placeholder="e.g., Custom Voice",
                    help="Name for this custom voice",
                    key="mp3_clone_name"
                )
            
            if uploaded_voice_file and mp3_speaker_name:
                st.info("⚠️ Best results with clear, single-speaker audio between 3-30 seconds.")
                
                # Display file information
                file_size_mb = len(uploaded_voice_file.read()) / (1024 * 1024)
                uploaded_voice_file.seek(0)  # Reset file pointer
                st.write(f"📁 **File:** {uploaded_voice_file.name} ({file_size_mb:.1f} MB)")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Clone Voice from MP3", key="clone_mp3_voice_btn"):
                        try:
                            from voice_editor import clone_voice_from_audio_file
                            
                            # Save uploaded file temporarily
                            temp_audio_path = temp_dir / f"uploaded_voice_{mp3_speaker_name.replace(' ', '_')}.{uploaded_voice_file.name.split('.')[-1]}"
                            with open(temp_audio_path, "wb") as f:
                                f.write(uploaded_voice_file.read())
                            
                            with st.spinner(f"Cloning voice from {uploaded_voice_file.name}..."):
                                success = clone_voice_from_audio_file(
                                    audio_file_path=temp_audio_path, 
                                    speaker_name=mp3_speaker_name
                                )
                            
                            if success:
                                st.success(f"✅ Voice '{mp3_speaker_name}' cloned successfully from MP3!")
                                # Add to available voices
                                voice_id = f"cloned_{mp3_speaker_name.lower().replace(' ', '_')}"
                                if voice_id not in st.session_state.available_voices:
                                    st.session_state.available_voices.append(voice_id)
                                st.rerun()
                            else:
                                st.error("❌ Voice cloning from MP3 failed.")
                                
                        except ImportError:
                            st.error("❌ Install required packages: `pip install pydub librosa`")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            st.info("💡 Make sure your audio file has clear speech and is not too long.")
                
                with col2:
                    if st.button("🎵 Test Voice", key="test_mp3_cloned_voice"):
                        # Test the voice that would be cloned
                        test_text = "Hello! This is a preview of how the cloned voice will sound from your uploaded audio file."
                        
                        # Check if this voice is already cloned
                        cloned_voices = st.session_state.get('cloned_voices', {})
                        if mp3_speaker_name in cloned_voices:
                            st.info(f"🎵 Voice '{mp3_speaker_name}' already exists! Testing cloned voice...")
                            
                            try:
                                from voice_editor import generate_voice_preview
                                
                                # Create voice config for the cloned voice - FORCE XTTS-v2
                                voice_config = {
                                    mp3_speaker_name: {
                                        "model": "tts_models/multilingual/multi-dataset/xtts_v2",
                                        "type": "cloned",
                                        "speaker_wav": cloned_voices[mp3_speaker_name]['speaker_wav'],
                                        "description": f"🎆 XTTS-v2 cloned voice from uploaded MP3"
                                    }
                                }
                                
                                st.info("🎵 Testing with XTTS-v2 voice cloning model...")
                                generate_voice_preview(test_text, mp3_speaker_name, voice_config)
                            
                            except Exception as e:
                                st.error(f"❌ Failed to test cloned voice: {e}")
                                st.info("💡 Try cloning the voice again if the test fails.")
                        else:
                            st.warning("⚠️ Voice not cloned yet. Clone the voice first, then test it.")
                
                # Show audio preview
                if uploaded_voice_file:
                    st.write("**🔊 Preview Uploaded Audio:**")
                    st.audio(uploaded_voice_file)
        
        # Show available voices
        st.write("**Available Voices:**")
        voice_cols = st.columns(min(len(st.session_state.available_voices), 4))
        for i, voice in enumerate(st.session_state.available_voices):
            with voice_cols[i % 4]:
                voice_display = voice.split('/')[-1] if '/' in voice else voice
                if voice.startswith('cloned_'):
                    st.success(f"🎬 {voice_display.replace('cloned_', '').replace('_', ' ').title()}")
                else:
                    st.info(f"🤖 {voice_display}")
        
        st.divider()
        
        # STEP 2: Speaker Voice Assignment
        st.subheader("🎭 Step 2: Assign Voices to Speakers")
        
        # Multi-speaker toggle
        use_multi_speaker = st.checkbox(
            "🎭 Multi-Speaker Mode",
            value=len(st.session_state.detected_speakers) > 1,
            help="Use different voices for different speakers"
        )
        
        speaker_voices = {}
        
        if use_multi_speaker and st.session_state.detected_speakers:
            st.write("Assign a voice to each detected speaker:")
            
            for i, (speaker_name, speaker_data) in enumerate(st.session_state.detected_speakers.items()):
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    st.write(f"**🎤 {speaker_name}**")
                    st.caption(f"{speaker_data.dialogue_count} dialogues")
                
                with col2:
                    assigned_voice = st.selectbox(
                        f"Voice for {speaker_name}",
                        options=st.session_state.available_voices,
                        index=i % len(st.session_state.available_voices),
                        key=f"voice_assignment_{speaker_name}",
                        label_visibility="collapsed"
                    )
                    speaker_voices[speaker_name] = assigned_voice
                
                with col3:
                    if st.button("🎵", key=f"preview_{speaker_name}", help="Preview voice"):
                        # Generate voice preview
                        preview_text = "Hello, this is a preview of your selected voice for this audiobook. The quality and tone will be consistent throughout all chapters."
                        
                        # Check if this is a cloned voice
                        is_cloned_voice = assigned_voice.startswith('cloned_')
                        cloned_voices = st.session_state.get('cloned_voices', {})
                        
                        if is_cloned_voice:
                            # Extract the original speaker name from the voice ID
                            original_speaker_name = assigned_voice.replace('cloned_', '').replace('_', ' ').title()
                            
                            # Find the matching cloned voice data
                            matching_cloned_voice = None
                            for cloned_name, cloned_data in cloned_voices.items():
                                if cloned_name.lower().replace(' ', '_') == assigned_voice.replace('cloned_', ''):
                                    matching_cloned_voice = (cloned_name, cloned_data)
                                    break
                            
                            if matching_cloned_voice:
                                cloned_name, cloned_data = matching_cloned_voice
                                try:
                                    from voice_editor import generate_voice_preview
                                    
                                    # Create voice config for the cloned voice
                                    voice_config = {
                                        cloned_name: {
                                            "model": "tts_models/multilingual/multi-dataset/xtts_v2",
                                            "type": "cloned",
                                            "speaker_wav": cloned_data['speaker_wav'],
                                            "description": f"🎆 Custom cloned voice from YouTube"
                                        }
                                    }
                                    
                                    with st.spinner(f"Generating cloned voice preview for {cloned_name}..."):
                                        generate_voice_preview(preview_text, cloned_name, voice_config)
                                    
                                except Exception as e:
                                    st.error(f"❌ Failed to generate cloned voice preview: {e}")
                                    st.info("💡 The cloned voice may not be properly configured.")
                            else:
                                st.error(f"❌ Cloned voice data not found for {original_speaker_name}")
                                st.info("🔄 Please try cloning the voice again.")
                        else:
                            # Regular voice preview
                            try:
                                from voice_editor import generate_voice_preview
                                
                                # Create a simple voice config for regular voices
                                if 'vctk' in assigned_voice:
                                    voice_config = {
                                        assigned_voice: {
                                            "model": assigned_voice,
                                            "type": "multi",
                                            "speaker": "p225",  # Default speaker
                                            "description": "Built-in multi-speaker voice"
                                        }
                                    }
                                else:
                                    voice_config = {
                                        assigned_voice: {
                                            "model": assigned_voice,
                                            "type": "single",
                                            "description": "Built-in single speaker voice"
                                        }
                                    }
                                
                                with st.spinner(f"Generating preview for {assigned_voice.split('/')[-1]}..."):
                                    generate_voice_preview(preview_text, assigned_voice, voice_config)
                                
                            except Exception as e:
                                st.error(f"❌ Preview failed: {e}")
                                st.info("💡 Try selecting a different voice model.")
        
        else:
            # Single voice mode
            st.write("Select a single voice for the entire audiobook:")
            single_voice = st.selectbox(
                "Voice Selection",
                options=st.session_state.available_voices,
                key="single_voice_selection"
            )
            speaker_voices = {"Narrator": single_voice}
        
        st.divider()
        
        # STEP 3: Output Settings & Generation
        st.subheader("⚙️ Step 3: Generate Audio & Video")
        
        # Audio/Video toggle
        output_type = st.radio(
            "Select Output Type:",
            options=["🎧 Audio Only", "🎬 Audio + Video"],
            horizontal=True
        )
        
        create_video = (output_type == "🎬 Audio + Video")
        
        # Video settings if creating video
        if create_video:
            if not st.session_state.extracted_images:
                st.warning("⚠️ No images found. Video will use text slides only.")
            else:
                st.info(f"✅ Using {len(st.session_state.extracted_images)} extracted images for video")
            
            col1, col2 = st.columns(2)
            with col1:
                video_quality = st.selectbox("Video Quality", ["720p", "1080p"], index=1)
                video_title = st.text_input("Video Title", value=st.session_state.book_metadata['title'])
            with col2:
                video_fps = st.selectbox("Frame Rate", ["24 fps", "30 fps"], index=0)
                video_author = st.text_input("Video Author", value=st.session_state.book_metadata['author'])
        
        st.divider()
        st.subheader("🚀 Step 4: Generate Audiobook")
        
        # Show summary
        with st.expander("📋 Generation Summary", expanded=True):
            st.write(f"**📚 Chapters:** {len(st.session_state.processed_chapters)}")
            st.write(f"**🎭 Speakers:** {len(speaker_voices)}")
            
            for speaker, voice in speaker_voices.items():
                voice_display = voice.split('/')[-1] if '/' in voice else voice
                if voice.startswith('cloned_'):
                    st.write(f"• **{speaker}**: 🎬 {voice_display.replace('cloned_', '').replace('_', ' ').title()} (Custom)")
                else:
                    st.write(f"• **{speaker}**: 🤖 {voice_display} (Built-in)")
            
            if create_video:
                st.write(f"**🎬 Video:** {video_quality} with {len(st.session_state.extracted_images)} images")
        
        # Generate button
        if st.button("🎧 Generate Audiobook", type="primary", use_container_width=True):
            st.info("🎯 **Individual Chapter Processing**: Each chapter will be processed and made available for download immediately.")
            
            # Initialize session state for tracking progress
            if 'audiobook_progress' not in st.session_state:
                st.session_state.audiobook_progress = {
                    'completed_chapters': [],
                    'failed_chapters': [],
                    'current_chapter': None,
                    'total_chapters': len(st.session_state.processed_chapters),
                    'completed_names': set(),
                }
            
            # Create output directories
            mp3_output_dir = temp_dir / "mp3_output"
            mp3_output_dir.mkdir(exist_ok=True)
            
            video_output_dir = None
            if create_video:
                video_output_dir = temp_dir / "video_output"
                video_output_dir.mkdir(exist_ok=True)
            
            # Progress tracking
            progress_placeholder = st.empty()
            downloads_placeholder = st.empty()
            
            # Process each chapter individually
            primary_voice = list(speaker_voices.values())[0]
            cloned_voices_data = st.session_state.get('cloned_voices', {})
            
            for i, chapter in enumerate(st.session_state.processed_chapters):
                chapter_name = chapter['name'].replace('.txt', '')
                st.session_state.audiobook_progress['current_chapter'] = chapter_name
                
                with progress_placeholder.container():
                    st.write(f"🔄 **Processing Chapter {i+1}/{len(st.session_state.processed_chapters)}: {chapter_name}**")
                    progress_bar = st.progress((i) / len(st.session_state.processed_chapters))
                
                # If this chapter was already completed in a previous run (e.g., after a download-triggered rerun), skip processing
                if 'completed_names' in st.session_state.audiobook_progress and chapter_name in st.session_state.audiobook_progress['completed_names']:
                    with progress_placeholder.container():
                        st.write(f"⏭️ Skipping already completed chapter: {chapter_name}")
                        progress_bar = st.progress((i + 1) / len(st.session_state.processed_chapters))
                    # Still refresh downloads list to show already completed items
                    downloads_placeholder.empty()
                    with downloads_placeholder.container():
                        st.subheader("📥 Available Downloads")
                        if st.session_state.audiobook_progress['completed_chapters']:
                            st.write(f"**✅ Completed Chapters ({len(st.session_state.audiobook_progress['completed_chapters'])}):**")
                            for completed in st.session_state.audiobook_progress['completed_chapters']:
                                col1, col2, col3 = st.columns([3, 2, 2])
                                with col1:
                                    st.write(f"**{completed['index']:02d}. {completed['name']}**")
                                with col2:
                                    audio_bytes = completed['audio_file'].read_bytes()
                                    st.download_button(
                                        "🎧 Audio",
                                        data=audio_bytes,
                                        file_name=f"{completed['name']}.mp3",
                                        mime="audio/mp3",
                                        key=completed.get('audio_key', f"download_audio_{completed['uid']}")
                                    )
                                with col3:
                                    if completed['video_file'] and completed['video_file'].exists():
                                        video_bytes = completed['video_file'].read_bytes()
                                        st.download_button(
                                            "🎬 Video",
                                            data=video_bytes,
                                            file_name=f"{completed['name']}.mp4",
                                            mime="video/mp4",
                                            key=completed.get('video_key', f"download_video_{completed['uid']}")
                                        )
                                    else:
                                        st.write("—")
                        if st.session_state.audiobook_progress['failed_chapters']:
                            st.write(f"**❌ Failed Chapters ({len(st.session_state.audiobook_progress['failed_chapters'])}):**")
                            for failed in st.session_state.audiobook_progress['failed_chapters']:
                                st.write(f"• **{failed['index']:02d}. {failed['name']}**: {failed['error']}")
                    continue
                
                try:
                    # Create temporary input for this chapter
                    chapter_input_dir = temp_dir / f"chapter_input_{i}"
                    chapter_input_dir.mkdir(exist_ok=True)
                    
                    chapter_file = chapter_input_dir / chapter['name']
                    chapter_file.write_text(chapter['content'], encoding='utf-8')
                    
                    # Generate audio for this chapter
                    chapter_mp3_dir = temp_dir / f"chapter_mp3_{i}"
                    chapter_mp3_dir.mkdir(exist_ok=True)
                    
                    def chapter_progress(msg):
                        with progress_placeholder.container():
                            st.write(f"🔄 **Chapter {i+1}/{len(st.session_state.processed_chapters)}: {chapter_name}**")
                            st.write(f"   {msg}")
                            progress_bar = st.progress((i + 0.5) / len(st.session_state.processed_chapters))
                    
                    convert_txt_to_mp3(
                        input_dir=chapter_input_dir,
                        output_dir=chapter_mp3_dir,
                        model_name=primary_voice,
                        progress_callback=chapter_progress,
                        cloned_voices_data=cloned_voices_data,
                        xtts_settings=st.session_state.get('xtts_settings') if hasattr(st, 'session_state') else None,
                    )
                    
                    # Find the generated audio file
                    mp3_files = list(chapter_mp3_dir.glob("*.mp3"))
                    if mp3_files:
                        chapter_mp3_file = mp3_files[0]
                        final_mp3_path = mp3_output_dir / f"{chapter_name}.mp3"
                        
                        # Copy to final location
                        import shutil
                        shutil.copy2(chapter_mp3_file, final_mp3_path)
                        
                        # Create video if requested
                        chapter_video_file = None
                        if create_video and VideoExporter:
                            try:
                                video_exporter = VideoExporter()
                                if video_exporter.check_ffmpeg_available():
                                    chapter_progress("🎬 Creating video...")
                                    
                                    # Use images if available, otherwise text slides
                                    images_dir = None
                                    if st.session_state.extracted_images:
                                        images_dir = temp_dir / "video_images"
                                        images_dir.mkdir(exist_ok=True)
                                        for img_path in st.session_state.extracted_images:
                                            shutil.copy2(img_path, images_dir / img_path.name)
                                    
                                    videos = video_exporter.create_audiobook_video(
                                        chapters_audio_dir=chapter_mp3_dir,
                                        images_dir=images_dir,
                                        output_dir=video_output_dir,
                                        book_title=video_title if create_video else st.session_state.book_metadata['title'],
                                        author=video_author if create_video else st.session_state.book_metadata['author']
                                    )
                                    
                                    if videos:
                                        chapter_video_file = videos[0]
                                        
                            except Exception as video_error:
                                chapter_progress(f"⚠️ Video creation failed: {video_error}")
                        
                        # Add to completed chapters (avoid duplicates across reruns)
                        if chapter_name not in st.session_state.audiobook_progress['completed_names']:
                            entry_uid = uuid.uuid4().hex
                            st.session_state.audiobook_progress['completed_chapters'].append({
                                'uid': entry_uid,
                                'name': chapter_name,
                                'audio_file': final_mp3_path,
                                'video_file': chapter_video_file,
                                'index': i + 1,
                                'audio_key': f"download_audio_{entry_uid}",
                                'video_key': f"download_video_{entry_uid}",
                            })
                            st.session_state.audiobook_progress['completed_names'].add(chapter_name)
                        else:
                            # Update existing entry paths if needed
                            for entry in st.session_state.audiobook_progress['completed_chapters']:
                                if entry['name'] == chapter_name:
                                    entry['audio_file'] = final_mp3_path
                                    entry['video_file'] = chapter_video_file
                                    break
                        
                        with progress_placeholder.container():
                            st.write(f"✅ **Chapter {i+1} Complete: {chapter_name}**")
                            progress_bar = st.progress((i + 1) / len(st.session_state.processed_chapters))
                            
                    else:
                        raise Exception("No audio file generated")
                        
                except Exception as e:
                    st.session_state.audiobook_progress['failed_chapters'].append({
                        'name': chapter_name,
                        'error': str(e),
                        'index': i + 1
                    })
                    with progress_placeholder.container():
                        st.write(f"❌ **Chapter {i+1} Failed: {chapter_name}**")
                        st.write(f"   Error: {e}")
                        progress_bar = st.progress((i + 1) / len(st.session_state.processed_chapters))
                
                # Update progress display only - don't show download buttons during generation
                downloads_placeholder.empty()
                with downloads_placeholder.container():
                    st.subheader("📊 Generation Progress")
                    
                    # Show completed count
                    if st.session_state.audiobook_progress['completed_chapters']:
                        st.write(f"**✅ Completed: {len(st.session_state.audiobook_progress['completed_chapters'])} / {len(st.session_state.processed_chapters)} chapters**")
                        
                        # List completed chapters without download buttons
                        for completed in st.session_state.audiobook_progress['completed_chapters']:
                            st.write(f"• {completed['index']:02d}. {completed['name']} ✓")
                    
                    # Show failed chapters
                    if st.session_state.audiobook_progress['failed_chapters']:
                        st.write(f"**❌ Failed: {len(st.session_state.audiobook_progress['failed_chapters'])} chapters**")
                        for failed in st.session_state.audiobook_progress['failed_chapters']:
                            st.write(f"• {failed['index']:02d}. {failed['name']}: {failed['error']}")
            
            # Final summary and downloads
            with progress_placeholder.container():
                st.success(f"🎉 **Processing Complete!**")
                total_completed = len(st.session_state.audiobook_progress['completed_chapters'])
                total_failed = len(st.session_state.audiobook_progress['faile                st.write(f"• ✅ **{total_completed}** chapters completed successfully")
                if total_failed > 0:
                    st.write(f"• ❌ **{total_failed}** chapters failed")
            
            # Show all downloads at the end - this won't interrupt the process
            if st.session_state.audiobook_progress['completed_chapters']:
                downloads_placeholder.empty()
                with downloads_placeholder.container():
                    st.subheader("📥 All Downloads Available")
                    st.info("✨ All chapters have been generated! You can now download them without interrupting the process.")
                    
                    # Create tabs for better organization
                    audio_tab, video_tab = st.tabs(["🎧 Audio Files", "🎬 Video Files"])
                    
                    with audio_tab:
                        st.write("**Download Audio Chapters:**")
                        for completed in st.session_state.audiobook_progress['completed_chapters']:
                            col1, col2 = st.columns([4, 2])
                            with col1:
                                st.write(f"**{completed['index']:02d}. {completed['name']}**")
                            with col2:
                                audio_bytes = completed['audio_file'].read_bytes()
                                st.download_button(
                                    "⬇️ Download MP3",
                                    data=audio_bytes,
                                    file_name=f"{completed['name']}.mp3",
                                    mime="audio/mp3",
                                    key=f"final_audio_{completed['uid']}"
                                )
                    
                    with video_tab:
                        video_chapters = [c for c in st.session_state.audiobook_progress['completed_chapters'] 
                                        if c.get('video_file') and c['video_file'].exists()]
                        if video_chapters:
                            st.write("**Download Video Chapters:**")
                            for completed in video_chapters:
                                col1, col2 = st.columns([4, 2])
                                with col1:
                                    st.write(f"**{completed['index']:02d}. {completed['name']}**")
                                with col2:
                                    video_bytes = completed['video_file'].read_bytes()
                                    st.download_button(
                                        "⬇️ Download MP4",
                                        data=video_bytes,
                                        file_name=f"{completed['name']}.mp4",
                                        mime="video/mp4",
                                        key=f"final_video_{completed['uid']}"
                                    )
                        else:
                            st.write("No videos were generated.")
            
            # Create combined audiobook if multiple chapters
st.divider()
st.markdown("**💡 Workflow:** PDF Processing & Speaker Detection → Audio & Video Creation with Voice Editor")
st.markdown("*Transform any PDF into a professional audiobook with advanced features like speaker detection, voice cloning, and video export - all in one streamlined interface.*")
