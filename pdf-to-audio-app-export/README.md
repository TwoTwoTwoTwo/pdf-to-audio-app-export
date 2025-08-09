# 📘 PDF to Audiobook Converter

**Transform PDFs into professional audiobooks with AI-powered speaker detection, voice cloning, and video export capabilities.**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![TTS](https://img.shields.io/badge/TTS-0.22+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🌟 **Key Features**

### 📄 **Advanced PDF Processing**
- **Smart Text Extraction**: Extract clean, readable text with advanced paragraph reconstruction
- **Intelligent Image Filtering**: Extract only actual figures/illustrations, not page scans
- **Enhanced Metadata Extraction**: Accurate title and author detection with multiple fallback methods
- **Multilingual Support**: Detect and process content in multiple languages
- **Chapter Organization**: Automatic chapter detection and individual file creation

### 🎤 **Professional TTS & Voice Features**
- **High-Quality Vocoders**: Premium HiFiGAN-v2, FastPitch, and VCTK multi-speaker models
- **XTTS-v2 Voice Cloning**: Clone voices from YouTube videos or audio files (17+ languages)
- **Multi-Speaker Detection**: Automatically detect and assign different voices to characters
- **Enhanced Audio Processing**: Automatic normalization, noise reduction, and quality enhancement
- **Real Voice Previews**: Test any voice with actual text before generation

### 🎬 **Video & Export Features**
- **Video Slideshow Creation**: Generate MP4 videos with synchronized audio and images
- **Multiple Output Formats**: Audio-only or audio+video with customizable quality settings
- **Batch Processing**: Process multiple chapters simultaneously
- **Professional Export**: ZIP downloads with organized file structure

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd pdf-to-audio-app
```

### 2. Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install System Dependencies

#### macOS
```bash
brew install ffmpeg tesseract
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg tesseract-ocr
```

#### Windows
1. **FFmpeg**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. **Tesseract**: Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

Make sure both are added to your system PATH.

### 4. Check Dependencies (Recommended)

```bash
# Check if everything is set up correctly
python3 setup.py

# Or run the complete setup (creates venv + installs dependencies)
python3 setup.py --install
```

### 5. Run the Application

```bash
# Quick start (recommended)
python3 start.py

# Or run directly
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## ✨ Features

### Core Features
- **PDF Text Extraction**: Extract clean, readable text from PDFs
- **Image Extraction**: Extract figures and images for video creation
- **Advanced Text Cleaning**: Remove page numbers, headers, and marginalia
- **Chapter Organization**: Automatically organize content into chapters

### Advanced Features
- **🎭 Speaker Detection**: Automatically detect multiple speakers in dialogue-heavy texts
- **🎬 Voice Cloning**: Clone voices from YouTube videos using Coqui TTS
- **🎵 High-Quality TTS**: Generate audio using advanced text-to-speech models
- **📹 Video Export**: Create video slideshows with synchronized audio
- **👥 Multi-Speaker Support**: Assign different voices to different speakers

## 🎯 How to Use

### Basic Workflow

1. **Upload PDF**: Use the web interface to upload your PDF file
2. **Extract Content**: The app will automatically extract text and images
3. **Review Results**: Check extracted chapters and detected speakers
4. **Generate Audio**: Choose voices and generate audiobook files
5. **Create Video** (Optional): Generate video slideshow with images

### Advanced Options

- **Custom Voice Cloning**: Clone voices from YouTube videos
- **Speaker Assignment**: Assign specific voices to different characters
- **Video Creation**: Generate MP4 videos with images and audio
- **Batch Processing**: Process multiple chapters simultaneously

## 📁 Output Structure

```
output/
├── chapters/                    # Individual chapter text files
│   ├── chapter_01_Introduction.txt
│   └── chapter_02_Methods.txt
├── audio/                      # Generated audio files
│   ├── chapter_01.mp3
│   └── chapter_02.mp3
├── images/                     # Extracted images
│   ├── fig_001.jpg
│   └── fig_002.png
├── video/                      # Generated videos (optional)
│   └── audiobook_video.mp4
└── metadata/                   # Processing metadata
    └── speaker_manifest.json
```

## ⚙️ Configuration

### Voice Models

The app supports multiple TTS engines:
- **Coqui TTS**: High-quality neural text-to-speech
- **Custom Voices**: Clone voices from audio samples
- **Multi-Speaker**: Different voices for different characters

### Quality Settings

- **Audio Quality**: Configure bitrate and sample rate
- **Video Quality**: Choose between 720p and 1080p output
- **Processing Speed**: Balance between quality and speed

## 🛠️ Helper Scripts

The repository includes several helper scripts to make setup and usage easier:

### `setup.py` - Dependency Checker
```bash
# Run health check
python3 setup.py

# Complete setup with virtual environment
python3 setup.py --install
```

This script:
- ✅ Checks Python version compatibility
- ✅ Verifies system dependencies (FFmpeg, Tesseract)
- ✅ Checks Python package installations
- 🔧 Creates virtual environment (with --install)
- 📦 Installs all requirements (with --install)

### `start.py` - Quick Launcher
```bash
# Start app with dependency check
python3 start.py

# Start app without dependency check
python3 start.py --skip-check
```

This script:
- 🔍 Runs a quick dependency check
- 🚀 Starts the Streamlit application
- 🌐 Opens browser automatically
- ⚡ Provides helpful error messages if something goes wrong

## 🔧 Recent Fixes & Improvements

### YouTube Voice Cloning Fixes

We've implemented comprehensive fixes for YouTube download robustness:

- **🔄 Multi-strategy download approach** with 4 fallback methods
- **📡 Enhanced error handling** for partial downloads and connection issues
- **🔁 Automatic retry logic** with chunk-based downloading
- **🌐 Geo-bypass and proxy support** for restricted content
- **🔇 Warning suppression** for cleaner output

**Test your setup:**
```bash
# Run comprehensive tests
python3 test_youtube_and_warnings.py
```

**For detailed troubleshooting, see:** [YOUTUBE_DOWNLOAD_FIXES.md](./YOUTUBE_DOWNLOAD_FIXES.md)

## 🔧 Troubleshooting

### Common Issues

**"FFmpeg not found"**
```bash
# Verify FFmpeg installation
ffmpeg -version

# If not installed, follow system dependency instructions above
```

**"Tesseract not found"**
```bash
# Verify Tesseract installation
tesseract --version

# If not installed, follow system dependency instructions above
```

**"TTS model download failed"**
- Check internet connection
- Models are downloaded automatically on first use
- Large models may take time to download

**"Voice cloning failed"**
- Ensure YouTube URL is accessible
- Check audio quality of source video
- Only use content you have permission to use

### Performance Tips

- Use shorter text chunks for faster processing
- GPU acceleration available for TTS models (install PyTorch with CUDA)
- Close other applications to free up memory for large PDFs

## 📋 System Requirements & Dependencies

### System Requirements

#### Minimum Requirements
- **RAM**: 4GB (8GB recommended for voice cloning)
- **Storage**: 2GB free space for models and temp files
- **CPU**: Multi-core processor recommended
- **Internet**: Required for initial model downloads

#### Recommended Specifications
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster TTS)
- **Storage**: SSD with 10GB+ free space

### Complete Dependencies List

#### System Dependencies (Required)
```bash
# macOS
brew install ffmpeg tesseract

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg tesseract-ocr

# Windows
# Download and install manually:
# - FFmpeg: https://ffmpeg.org/download.html
# - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
```

#### Python Dependencies (Core)
```bash
# Web Framework
streamlit>=1.28.0

# PDF Processing
pdfplumber>=0.7.0          # PDF text extraction
PyMuPDF>=1.23.0            # PDF processing and image extraction
Pillow>=9.5.0              # Image processing
pytesseract>=0.3.10        # OCR for image filtering

# Text Processing
langdetect>=1.0.9          # Language detection
unicodedata2>=15.0.0       # Unicode normalization
charset-normalizer>=3.2.0  # Text encoding handling

# Data Handling
pandas>=2.0.0             # Data manipulation
numpy>=1.24.0            # Numerical computing
requests>=2.31.0          # HTTP requests
tqdm>=4.65.0              # Progress bars
packaging>=23.1           # Package utilities
```

#### TTS & Audio Dependencies
```bash
# Core TTS Engine
TTS>=0.22.0               # Coqui TTS for voice synthesis

# PyTorch (AI/ML Framework)
torch>=2.0.0              # Deep learning framework
torchaudio>=2.0.0         # Audio processing for PyTorch

# Audio Processing
pydub>=0.25.1             # Audio manipulation
librosa>=0.10.0           # Audio analysis
```

#### Voice Cloning Dependencies (Optional)
```bash
# YouTube Integration
yt-dlp>=2023.7.6           # YouTube video/audio downloading
youtube-transcript-api>=0.6.1  # YouTube caption extraction
```

#### GPU Support Dependencies (Optional)
```bash
# For CUDA 11.8 (faster TTS processing)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Performance boost (optional)
numba>=0.57.0             # JIT compilation for faster audio processing
```

#### Development Dependencies (Optional)
```bash
# Testing and development
pytest>=7.0.0             # Testing framework
black>=22.0.0             # Code formatting
flake8>=4.0.0             # Code linting
```

### TTS Model Information

The application will automatically download the following TTS models on first use:

#### High-Quality Models (Recommended)
- **VCTK Multi-Speaker** (~400MB): 20+ English voices with different characteristics
- **GlowTTS** (~100MB): High-quality single speaker model
- **FastPitch** (~80MB): Fast, high-quality synthesis

#### Voice Cloning Models (Optional)
- **XTTS-v2** (~1.8GB): State-of-the-art voice cloning, 17+ languages
- **YourTTS** (~500MB): Fallback voice cloning model

#### Legacy Models (Fallback)
- **Tacotron2-DDC** (~200MB): Reliable fallback model

### Installation Commands Summary

```bash
# Complete installation (recommended)
git clone <your-repo-url>
cd pdf-to-audio-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install system dependencies
# macOS:
brew install ffmpeg tesseract

# Ubuntu/Debian:
sudo apt install ffmpeg tesseract-ocr

# Windows: Download and install FFmpeg and Tesseract manually

# Run dependency check
python3 setup.py

# Start the application
python3 start.py
```

### Optional Enhancements

```bash
# For better performance (optional)
pip install numba

# For YouTube voice cloning (optional)
pip install yt-dlp youtube-transcript-api

# For GPU acceleration (optional, requires NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Legal Notice

- Only use voice cloning with content you have permission to use
- Respect copyright when processing PDFs
- Generated voices should not be used to impersonate others without consent

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Look through existing GitHub issues
3. Create a new issue with:
   - Your operating system
   - Python version
   - Error message
   - Steps to reproduce

## 🚀 Technical Improvements & Advanced Features

### 📄 **Enhanced PDF Processing**
- **Smart Title Extraction**: Multiple pattern matching strategies with priority scoring
- **Advanced Image Filtering**: OCR-based text detection to filter out page scans
- **Multilingual Content Detection**: Chunk-based analysis for better language detection
- **Paragraph Reconstruction**: Intelligent joining of text split across page breaks
- **Marginalia Removal**: Advanced pattern matching for headers, footers, and page numbers

### 🎤 **TTS Engine Enhancements**
- **PyTorch Compatibility**: Automatic fixes for PyTorch 2.6+ loading issues
- **Model Fallback System**: Intelligent fallback from XTTS-v2 to VCTK when needed
- **Voice Preview System**: Real voice samples using actual text content
- **Multi-Speaker Intelligence**: Automatic voice assignment based on speaker characteristics
- **Audio Quality Enhancement**: Automatic normalization and noise reduction

### 🎬 **Video & Workflow Improvements**
- **Integrated Video Creation**: Seamless audio+video generation in single workflow
- **Professional Export Structure**: Organized ZIP downloads with proper file naming
- **Progress Tracking**: Real-time progress indicators throughout processing
- **Error Recovery**: Robust error handling with helpful user feedback
- **Dependency Management**: Automated setup scripts with health checks

### 📊 **Key Improvements Summary**

| Feature | Before | After |
|---------|--------|-------|
| Voice Quality | ❌ Low-quality Tacotron | ✅ Premium VCTK + HiFiGAN |
| Voice Cloning | ❌ XTTS-v2 crashes | ✅ Intelligent fallback system |
| Video Creation | ❌ Separate workflow | ✅ Integrated in Step 3 |
| Voice Preview | ❌ "Coming soon" message | ✅ Real audio generation |
| Image Filtering | ❌ 744 page scans | ✅ Only actual figures |
| Setup Process | ❌ Manual dependency checking | ✅ Automated setup scripts |
| Error Handling | ❌ Cryptic error messages | ✅ Helpful troubleshooting |
| Documentation | ❌ Scattered across files | ✅ Unified comprehensive guide |

### 🤖 **AI & Machine Learning Features**
- **Speaker Detection**: NLP-based character identification in dialogue
- **Voice Cloning**: XTTS-v2 integration with YouTube audio extraction
- **Content Analysis**: Automatic formality scoring and speaking pattern detection
- **Language Processing**: Multi-language support with intelligent text normalization
- **Audio Enhancement**: ML-based audio quality improvements

## 🎉 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- TTS powered by [Coqui TTS](https://github.com/coqui-ai/TTS)
- PDF processing using [PyMuPDF](https://github.com/pymupdf/PyMuPDF) and [pdfplumber](https://github.com/jsvine/pdfplumber)
- Audio processing with [pydub](https://github.com/jiaaro/pydub)
- Voice cloning with [XTTS-v2](https://github.com/coqui-ai/TTS)
- Video processing with [FFmpeg](https://ffmpeg.org/)
- OCR capabilities with [Tesseract](https://github.com/tesseract-ocr/tesseract)
