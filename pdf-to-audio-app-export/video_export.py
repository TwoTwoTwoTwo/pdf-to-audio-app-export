#!/usr/bin/env python3
"""
Video Export Module
Creates video slideshows combining audio narration with extracted images
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
import shutil
from datetime import timedelta
import random
import re

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pydub import AudioSegment
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

class VideoExporter:
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_export_"))
        self.output_width = 1920
        self.output_height = 1080
        self.fps = 30
        self.default_image_duration = 5.0  # seconds per image
        
    def __del__(self):
        # Clean up temp directory
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def check_ffmpeg_available(self) -> bool:
        """Check if FFmpeg is available on the system"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds"""
        if not AUDIO_AVAILABLE:
            raise ImportError("pydub is required for audio processing. Install with: pip install pydub")
        
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert to seconds
        except Exception as e:
            print(f"⚠️ Warning: Could not get duration of {audio_path}: {e}")
            return 0.0
    
    def prepare_images_for_video(
        self, 
        image_paths: List[Path], 
        audio_duration: float,
        title: str = "",
        chapter_title: str = ""
    ) -> List[Path]:
        """Prepare and resize images for video, adding text overlays"""
        
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for image processing. Install with: pip install pillow")
        
        prepared_images = []
        images_dir = self.temp_dir / "prepared_images"
        images_dir.mkdir(exist_ok=True)
        
        # Calculate how long each image should be displayed
        if image_paths:
            image_duration = max(audio_duration / len(image_paths), 2.0)  # At least 2 seconds per image
        else:
            image_duration = self.default_image_duration
        
        print(f"📸 Preparing {len(image_paths)} images for video...")
        print(f"⏱️ Each image will be shown for {image_duration:.1f} seconds")
        
        # Create title slide if we have title information
        if title or chapter_title:
            title_image = self.create_title_slide(title, chapter_title)
            if title_image:
                prepared_images.append(title_image)
        
        for i, img_path in enumerate(image_paths):
            try:
                # Open and process image
                with Image.open(img_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to fit video dimensions while maintaining aspect ratio
                    processed_img = self.resize_image_for_video(img)
                    
                    # Add image number overlay
                    processed_img = self.add_image_overlay(processed_img, f"Figure {i+1}")
                    
                    # Save processed image
                    output_path = images_dir / f"image_{i+1:04d}.jpg"
                    processed_img.save(output_path, 'JPEG', quality=90)
                    prepared_images.append(output_path)
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not process image {img_path}: {e}")
                continue
        
        # If no images were successfully processed, create a default slide
        if not prepared_images or (title and len(prepared_images) == 1):
            default_image = self.create_default_slide("Audio Content")
            if default_image:
                prepared_images.append(default_image)
        
        print(f"✅ Prepared {len(prepared_images)} images for video")
        return prepared_images
    
    def resize_image_for_video(self, img: Image.Image) -> Image.Image:
        """Resize image to fit video dimensions while maintaining aspect ratio"""
        
        # Calculate scaling to fit within video dimensions
        scale_w = self.output_width / img.width
        scale_h = self.output_height / img.height
        scale = min(scale_w, scale_h)
        
        # Resize image
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create canvas with video dimensions
        canvas = Image.new('RGB', (self.output_width, self.output_height), (20, 20, 30))  # Dark background
        
        # Center the image on the canvas
        x_offset = (self.output_width - new_width) // 2
        y_offset = (self.output_height - new_height) // 2
        canvas.paste(resized_img, (x_offset, y_offset))
        
        return canvas
    
    def add_image_overlay(self, img: Image.Image, text: str) -> Image.Image:
        """Add text overlay to image"""
        
        try:
            draw = ImageDraw.Draw(img)
            
            # Try to load a font, fall back to default if not available
            try:
                font_size = 36
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text at bottom right
            x = self.output_width - text_width - 20
            y = self.output_height - text_height - 20
            
            # Draw text with shadow for better visibility
            shadow_offset = 2
            draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 128))  # Shadow
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))  # Main text
            
        except Exception as e:
            print(f"⚠️ Warning: Could not add overlay text: {e}")
        
        return img
    
    def create_title_slide(self, title: str, chapter_title: str = "") -> Optional[Path]:
        """Create a title slide for the video"""
        
        try:
            # Create canvas
            canvas = Image.new('RGB', (self.output_width, self.output_height), (30, 40, 60))  # Blue background
            draw = ImageDraw.Draw(canvas)
            
            # Try to load fonts
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 72)
                chapter_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 48)
            except (OSError, IOError):
                title_font = ImageFont.load_default()
                chapter_font = ImageFont.load_default()
            
            # Draw title
            if title:
                # Center title
                bbox = draw.textbbox((0, 0), title, font=title_font)
                title_width = bbox[2] - bbox[0]
                title_height = bbox[3] - bbox[1]
                
                title_x = (self.output_width - title_width) // 2
                title_y = self.output_height // 3
                
                # Draw title with shadow
                draw.text((title_x + 3, title_y + 3), title, font=title_font, fill=(0, 0, 0, 128))
                draw.text((title_x, title_y), title, font=title_font, fill=(255, 255, 255))
            
            # Draw chapter title
            if chapter_title:
                bbox = draw.textbbox((0, 0), chapter_title, font=chapter_font)
                chapter_width = bbox[2] - bbox[0]
                
                chapter_x = (self.output_width - chapter_width) // 2
                chapter_y = title_y + title_height + 40 if title else self.output_height // 2
                
                # Draw chapter title with shadow
                draw.text((chapter_x + 2, chapter_y + 2), chapter_title, font=chapter_font, fill=(0, 0, 0, 128))
                draw.text((chapter_x, chapter_y), chapter_title, font=chapter_font, fill=(200, 220, 255))
            
            # Save title slide
            title_path = self.temp_dir / "title_slide.jpg"
            canvas.save(title_path, 'JPEG', quality=90)
            return title_path
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create title slide: {e}")
            return None
    
    def create_default_slide(self, text: str) -> Optional[Path]:
        """Create a default slide with just text"""
        
        try:
            canvas = Image.new('RGB', (self.output_width, self.output_height), (40, 40, 40))  # Dark gray
            draw = ImageDraw.Draw(canvas)
            
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 64)
            except (OSError, IOError):
                font = ImageFont.load_default()
            
            # Center text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (self.output_width - text_width) // 2
            y = (self.output_height - text_height) // 2
            
            # Draw text with shadow
            draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 128))
            draw.text((x, y), text, font=font, fill=(255, 255, 255))
            
            # Save slide
            slide_path = self.temp_dir / "default_slide.jpg"
            canvas.save(slide_path, 'JPEG', quality=90)
            return slide_path
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create default slide: {e}")
            return None
    
    def create_video_from_images_and_audio(
        self,
        image_paths: List[Path],
        audio_path: Path,
        output_path: Path,
        title: str = "",
        chapter_title: str = ""
    ) -> bool:
        """Create video by combining images and audio using FFmpeg"""
        
        if not self.check_ffmpeg_available():
            raise RuntimeError("FFmpeg is required for video creation. Please install FFmpeg.")
        
        # Get audio duration
        audio_duration = self.get_audio_duration(audio_path)
        if audio_duration == 0:
            print("❌ Could not determine audio duration")
            return False
        
        print(f"🎵 Audio duration: {timedelta(seconds=int(audio_duration))}")
        
        # Prepare images
        prepared_images = self.prepare_images_for_video(
            image_paths, audio_duration, title, chapter_title
        )
        
        if not prepared_images:
            print("❌ No images available for video creation")
            return False
        
        # Calculate timing for images
        image_duration = audio_duration / len(prepared_images)
        
        print(f"🎬 Creating video with {len(prepared_images)} images...")
        print(f"📏 Each image displayed for {image_duration:.1f} seconds")
        
        # Create file list for FFmpeg
        filelist_path = self.temp_dir / "filelist.txt"
        with open(filelist_path, 'w') as f:
            for img_path in prepared_images:
                f.write(f"file '{img_path.absolute()}'\n")
                f.write(f"duration {image_duration}\n")
            # Duplicate last image to ensure it displays for full duration
            f.write(f"file '{prepared_images[-1].absolute()}'\n")
        
        # Build FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-i', str(filelist_path),
            '-i', str(audio_path),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-r', str(self.fps),
            '-shortest',  # End when shortest stream ends
            '-movflags', '+faststart',  # Optimize for streaming
            str(output_path)
        ]
        
        try:
            print("⚙️ Running FFmpeg to create video...")
            print(f"   Command: {' '.join(ffmpeg_cmd[:8])}... (truncated)")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Video created successfully: {output_path}")
                return True
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Video creation timed out")
            return False
        except Exception as e:
            print(f"❌ Error creating video: {e}")
            return False
    
    def create_audiobook_video(
        self,
        chapters_audio_dir: Path,
        images_dir: Path,
        output_dir: Path,
        book_title: str = "",
        author: str = ""
    ) -> List[Path]:
        """Create video files for all audio chapters"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.m4a']:
            audio_files.extend(list(chapters_audio_dir.glob(ext)))
        
        if not audio_files:
            print(f"❌ No audio files found in {chapters_audio_dir}")
            return []
        
        audio_files.sort()
        
        # Find image files
        image_files = []
        if images_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']:
                image_files.extend(list(images_dir.glob(ext)))
            image_files.sort()
        
        print(f"🎵 Found {len(audio_files)} audio files")
        print(f"🖼️ Found {len(image_files)} image files")
        
        if not image_files:
            print("⚠️ No images found, videos will contain text slides only")
        
        created_videos = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n📹 Creating video {i}/{len(audio_files)}: {audio_file.stem}")
            
            # Extract chapter info from filename
            chapter_match = re.search(r'chapter[_\s]*(\d+)', audio_file.stem.lower())
            chapter_num = chapter_match.group(1) if chapter_match else str(i)
            
            # Create chapter title
            chapter_title = f"Chapter {chapter_num}"
            if len(audio_file.stem) > 10:
                # Use part of filename as title
                clean_name = re.sub(r'chapter[_\s]*\d+[_\s]*', '', audio_file.stem, flags=re.I)
                clean_name = clean_name.replace('_', ' ').title()
                if clean_name:
                    chapter_title += f": {clean_name[:50]}"
            
            # Distribute images across chapters
            if image_files:
                images_per_chapter = max(1, len(image_files) // len(audio_files))
                start_idx = i * images_per_chapter
                end_idx = start_idx + images_per_chapter
                chapter_images = image_files[start_idx:end_idx]
                
                # If last chapter, include remaining images
                if i == len(audio_files):
                    chapter_images.extend(image_files[end_idx:])
            else:
                chapter_images = []
            
            # Create video for this chapter
            video_filename = f"{chapter_num:02d}_{audio_file.stem}.mp4"
            video_path = output_dir / video_filename
            
            success = self.create_video_from_images_and_audio(
                image_paths=chapter_images,
                audio_path=audio_file,
                output_path=video_path,
                title=book_title,
                chapter_title=chapter_title
            )
            
            if success:
                created_videos.append(video_path)
                print(f"✅ Created: {video_path}")
            else:
                print(f"❌ Failed to create video for {audio_file}")
        
        print(f"\n🎉 Created {len(created_videos)} videos in {output_dir}")
        return created_videos
    
    def create_playlist_file(self, video_files: List[Path], output_dir: Path) -> Path:
        """Create a playlist file for the video series"""
        
        playlist_path = output_dir / "playlist.m3u8"
        
        with open(playlist_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            f.write("#EXT-X-PLAYLIST-TYPE:VOD\n")
            
            for video_file in sorted(video_files):
                # Get video duration (approximate)
                try:
                    if AUDIO_AVAILABLE:
                        # Try to get duration from audio track
                        duration = self.get_audio_duration(video_file)
                    else:
                        duration = 300  # Default 5 minutes
                    
                    f.write(f"#EXTINF:{duration:.0f},{video_file.stem}\n")
                    f.write(f"{video_file.name}\n")
                    
                except Exception:
                    f.write(f"#EXTINF:300,{video_file.stem}\n")
                    f.write(f"{video_file.name}\n")
            
            f.write("#EXT-X-ENDLIST\n")
        
        return playlist_path

def create_audiobook_videos(
    audio_dir: str,
    images_dir: str = None,
    output_dir: str = "audiobook_videos",
    title: str = "",
    author: str = ""
) -> bool:
    """Convenience function to create audiobook videos"""
    
    exporter = VideoExporter()
    
    try:
        audio_path = Path(audio_dir)
        images_path = Path(images_dir) if images_dir else None
        
        videos = exporter.create_audiobook_video(
            chapters_audio_dir=audio_path,
            images_dir=images_path,
            output_dir=Path(output_dir),
            book_title=title,
            author=author
        )
        
        if videos:
            # Create playlist
            playlist = exporter.create_playlist_file(videos, Path(output_dir))
            print(f"📋 Playlist created: {playlist}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Video creation failed: {e}")
        return False

def main():
    """Command line interface for video export"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create videos from audiobook chapters and images")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("--images", "-i", help="Directory containing images")
    parser.add_argument("--output", "-o", default="audiobook_videos", help="Output directory")
    parser.add_argument("--title", "-t", default="", help="Book title")
    parser.add_argument("--author", "-a", default="", help="Author name")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not PIL_AVAILABLE:
        print("❌ PIL/Pillow is required. Install with: pip install pillow")
        return 1
    
    if not AUDIO_AVAILABLE:
        print("❌ pydub is required. Install with: pip install pydub")
        return 1
    
    exporter = VideoExporter()
    if not exporter.check_ffmpeg_available():
        print("❌ FFmpeg is required. Please install FFmpeg and ensure it's in your PATH.")
        print("   - macOS: brew install ffmpeg")
        print("   - Ubuntu: sudo apt install ffmpeg")
        print("   - Windows: Download from https://ffmpeg.org/")
        return 1
    
    # Create videos
    success = create_audiobook_videos(
        audio_dir=args.audio_dir,
        images_dir=args.images,
        output_dir=args.output,
        title=args.title,
        author=args.author
    )
    
    if success:
        print(f"\n🎉 Audiobook videos created successfully!")
        print(f"📁 Output directory: {args.output}")
        return 0
    else:
        print("\n❌ Video creation failed")
        return 1

if __name__ == "__main__":
    exit(main())
