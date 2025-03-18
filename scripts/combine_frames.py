import os
import subprocess

# Path to your images folder
image_folder = 'movie_files/hire_shark'

# Ensure the images are sorted in the right order
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])

# Check if images are found
if not image_files:
    raise ValueError("No image files found in the specified folder.")


# Create the input pattern for FFmpeg
input_pattern = os.path.join(image_folder, 'my_fig%04d.png')  # Change to match your file naming (e.g., %03d for 3-digit numbers)

# Define output file name
output_file = 'shark_hire.mov'

# Run FFmpeg command to create a video
ffmpeg_command = [
    'ffmpeg',
    '-framerate', '11',  # Frame rate (can be adjusted)
    '-i', input_pattern,  # Input image pattern
    '-c:v', 'prores_ks',  # Codec for .mov format (ProRes)
    '-pix_fmt', 'yuv422p10le',  # Pixel format (adjust based on your needs)
    output_file
]

# Execute the command
subprocess.run(ffmpeg_command)

print(f"Video created successfully: {output_file}")
