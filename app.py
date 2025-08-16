import threading
import torch

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")


import sys
print("Python path:", sys.executable)
print("Python version:", sys.version)
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("PyTorch location:", torch.__file__)
except ImportError:
    print("PyTorch not found in this environment")

from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import urllib.request
import asyncio
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import edge_tts
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/output', exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs(os.path.join('Wav2Lip', 'checkpoints'), exist_ok=True)

WAV2LIP_MODEL = os.path.join('Wav2Lip', 'checkpoints', 'wav2lip_gan.pth')
AUDIO_FILE = "static/output/generated_audio.mp3"
VIDEO_OUTPUT = "static/output/output_video.mp4"
FINAL_VIDEO = "static/output/final_video.mp4"

print("System PATH:", os.environ.get('PATH'))

# Helper: robustly resolve ffmpeg/ffprobe executables on Windows
def _resolve_media_tool(exe_basename):
    try:
        if os.name == 'nt':
            candidates = [
                os.path.join(os.getcwd(), f"{exe_basename}.exe"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{exe_basename}.exe"),
                r"C:\\ffmpeg\\bin\\%s.exe" % exe_basename,
                r"D:\\ffmpeg\\bin\\%s.exe" % exe_basename,
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            return f"{exe_basename}.exe"
        return exe_basename
    except Exception:
        return exe_basename

def get_ffmpeg_path():
    return _resolve_media_tool('ffmpeg')

def get_ffprobe_path():
    return _resolve_media_tool('ffprobe')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_trending_topics(country='in', category='general'):
    try:
        url = "https://gnews.io/api/v4/top-headlines"
        params = {
            'token': GNEWS_API_KEY,
            'country': country,
            'lang': 'en',
            'max': 20,
            'category': category
        }
        response = requests.get(url, params=params)
        data = response.json()
        print("GNews response:", data)
        if response.status_code == 200 and "articles" in data:
            return [article["title"] for article in data["articles"][:20]]
        else:
            return ["No trending topics available"]
    except Exception as e:
        return [f"Error fetching trends: {e}"]
    
def fetch_trend_news(trend, country='in'):
    try:
        url = "https://gnews.io/api/v4/search"
        # Clean and encode the query to avoid syntax errors
        clean_trend = trend.strip().replace('"', '').replace("'", '')
        
        # If the trend is too complex, try with a simpler query
        if len(clean_trend.split()) > 3:
            # Take first 2-3 words for a simpler search
            words = clean_trend.split()[:3]
            clean_trend = ' '.join(words)
        
        params = {
            'token': GNEWS_API_KEY,
            'q': clean_trend,
            'country': country,
            'lang': 'en',
            'max': 15
        }
        response = requests.get(url, params=params)
        data = response.json()
        print("GNews response:", data)
        
        if response.status_code == 200 and "articles" in data:
            return [article["title"] + " - " + (article.get("description") or "") for article in data["articles"][:15]]
        elif "errors" in data:
            # If there's a syntax error, try with just the first word
            first_word = clean_trend.split()[0] if clean_trend.split() else clean_trend
            params['q'] = first_word
            response = requests.get(url, params=params)
            data = response.json()
            if response.status_code == 200 and "articles" in data:
                return [article["title"] + " - " + (article.get("description") or "") for article in data["articles"][:15]]
            else:
                return ["No related news found"]
        else:
            return ["No related news found"]
    except Exception as e:
        return [f"Error fetching news: {e}"]
    
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    if score["compound"] >= 0.05:
        return "positive"
    elif score["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Generate video script using AI
# MODIFIED: Added desired_tone parameter
def generate_script(trend, news, sentiment_analysis_result, desired_tone):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    # Base context
    context = f"Trend: {trend}\nRelated News:\n{' '.join(news)}"

    # Add sentiment analysis result if applicable
    if sentiment_analysis_result:
        context += f"\nOverall News Sentiment Analysis: {sentiment_analysis_result}"

    # Construct the prompt based on desired_tone
    if desired_tone and desired_tone != "neutral":
        # Specific tone requested by user
        prompt_instruction = f"Create a concise, conversational 4-minute news script that directly presents the information with an **{desired_tone}** tone."
        system_content_tone_part = f"The script should adopt an {desired_tone} tone. "
    else:
        # No specific tone or neutral selected, rely on general conversational tone
        prompt_instruction = "Create a concise, conversational 4-minute news script that directly presents the information."
        system_content_tone_part = "The tone should be conversational. "


    prompt = f"{context}\n{prompt_instruction} The script should be written as a continuous monologue without any scene directions, character changes, or technical instructions. Focus only on the actual spoken content."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a news script writer. Write natural, flowing scripts that can be read in 4 minutes. {system_content_tone_part}Do not include any directions, scene changes, or speaker labels. Write only the actual spoken content."
            },
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

# Update the generate_content route
# MODIFIED: Fetched 'sentiment' (which is now desired_tone) from request.form
@app.route('/generate_content', methods=['POST'])
def generate_content():
    try:
        selected_trend = request.form.get('trend')
        desired_tone = request.form.get('sentiment') # Fetch the user's selected sentiment/tone
        if not selected_trend:
            return jsonify({'error': 'No trend selected'}), 400

        # Fetch news
        news = fetch_trend_news(selected_trend)
        if not news:
            return jsonify({'error': 'No news found for the selected trend'}), 400

        # Perform sentiment analysis for additional context, but prioritize user's desired_tone
        sentiment_analysis_result = analyze_sentiment(' '.join(news))

        # Pass both the analysis result and the user's desired_tone to generate_script
        script = generate_script(selected_trend, news, sentiment_analysis_result, desired_tone)

        return jsonify({
            'success': True,
            'script': script
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _chunk_text_for_tts(text, max_chars=3000):
    # Simple chunker that tries to break on sentence boundaries, then on spaces
    chunks = []
    remaining = text.strip()
    sentence_endings = ['.', '!', '?', '\n']
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        cut = max_chars
        # try to find last sentence ending within window
        window = remaining[:max_chars]
        last_end = max((window.rfind(sep) for sep in sentence_endings))
        if last_end > 0:
            cut = last_end + 1
        else:
            # fallback to last space
            space_idx = window.rfind(' ')
            if space_idx > 0:
                cut = space_idx
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    return [c for c in chunks if c]

# Convert text to speech using Edge-TTS
async def text_to_speech(script, output_file, voice_name="en-US-JennyNeural"):
    try:
        print("▶️ Entered text_to_speech()")
        chunks = _chunk_text_for_tts(script, max_chars=3000)
        with open(output_file, "wb") as f:
            for idx, chunk in enumerate(chunks, start=1):
                if not chunk.strip():
                    continue
                print(f"   Converting chunk {idx}/{len(chunks)}: {chunk[:80]}...")
                communicate = edge_tts.Communicate(chunk, voice_name)
                async for data in communicate.stream():
                    if data["type"] == "audio":
                        f.write(data["data"])
        print(f"✅ Audio saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error in text_to_speech: {e}")
        raise


def convert_text_to_speech(script, output_file, voice_name): # Added voice_name
    """
    Wrapper to run the async text_to_speech function in a separate thread
    with its own event loop.
    """
    def run_async_tts():
        # Create and manage a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the async function until it completes
            loop.run_until_complete(text_to_speech(script, output_file, voice_name))
        finally:
            # Ensure the loop is closed
            loop.close()

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Run the async function in a new thread and wait for it to complete
        thread = threading.Thread(target=run_async_tts)
        thread.start()
        # Allow more time for longer scripts (e.g., ~4 minutes speech)
        thread.join(timeout=600)

        if thread.is_alive():
            print("TTS thread timed out and is still running.")
            raise TimeoutError("Text-to-speech conversion timed out after 10 minutes.")

        print(f"Text-to-speech conversion completed: {output_file}")
    except Exception as e:
        print(f"Error in convert_text_to_speech: {e}")
        raise

# Generate lip-synced video using Wav2Lip
def generate_video(image_path, audio_path, output_video):
    try:
        wav2lip_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Wav2Lip')
        inference_path = os.path.join(wav2lip_dir, 'inference.py')

        # Create temp directory inside Wav2Lip
        os.makedirs(os.path.join(wav2lip_dir, 'temp'), exist_ok=True)

        # Copy FFmpeg executables to Wav2Lip directory
        for exe in ['ffmpeg.exe', 'ffprobe.exe', 'ffplay.exe']:
            src = os.path.join(os.getcwd(), exe)
            dst = os.path.join(wav2lip_dir, exe)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get the current Python executable path
        python_executable = sys.executable

        # Update the checkpoint path to use Wav2Lip directory
        checkpoint_path = os.path.join(wav2lip_dir, 'checkpoints', 'wav2lip_gan.pth')

        # Convert all paths to absolute paths
        image_path = os.path.abspath(image_path)
        audio_path = os.path.abspath(audio_path)
        output_video = os.path.abspath(output_video)

        # Print debug information
        print("Debug Info:")
        print(f"Working Directory: {os.getcwd()}")
        print(f"Wav2Lip Directory: {wav2lip_dir}")
        print(f"Inference Path: {inference_path}")
        print(f"Checkpoint Path: {checkpoint_path}")
        print(f"Image Path: {image_path}")
        print(f"Audio Path: {audio_path}")
        print(f"Output Video Path: {output_video}")

        # Verify files exist
        print("\nFile Existence Checks:")
        print(f"Wav2Lip Directory exists: {os.path.exists(wav2lip_dir)}")
        print(f"Inference script exists: {os.path.exists(inference_path)}")
        print(f"Checkpoint file exists: {os.path.exists(checkpoint_path)}")
        print(f"Input image exists: {os.path.exists(image_path)}")
        print(f"Input audio exists: {os.path.exists(audio_path)}")

        # Add current directory to PATH
        current_dir = os.getcwd()
        os.environ['PATH'] = current_dir + os.pathsep + os.environ.get('PATH', '')

        # Add Wav2Lip directory to PATH as well
        env = os.environ.copy()
        env['PATH'] = wav2lip_dir + os.pathsep + env['PATH']
        env['CUDA_LAUNCH_BLOCKING'] = '1'

        # Construct command with additional parameters for stability
        command = [
            python_executable,
            inference_path,
            "--checkpoint_path", checkpoint_path,
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", output_video,
            "--resize_factor", "2",  # Add resize factor to prevent CUDA memory issues
            "--wav2lip_batch_size", "128",  # Adjust batch size
            "--nosmooth"  # Disable smoothing for better stability
        ]

        print("\nExecuting command:")
        print(" ".join(command))

        # Run the command with output capture and environment variables
        # Keep working directory as current_dir to maintain path consistency
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=current_dir
        )

        # Print any output
        if process.stdout:
            print("\nCommand output:")
            print(process.stdout)

        if process.stderr:
            print("\nCommand errors:")
            print(process.stderr)

        # Verify the video file was actually created
        if os.path.exists(output_video):
            print("✅ Video generated successfully at:", output_video)
        else:
            print("❌ Video file not found after generation at:", output_video)
            # Check if it exists in the Wav2Lip directory
            wav2lip_output = os.path.join(wav2lip_dir, os.path.basename(output_video))
            if os.path.exists(wav2lip_output):
                print(f"Video found in Wav2Lip directory, copying to: {output_video}")
                import shutil
                shutil.copy2(wav2lip_output, output_video)
            else:
                raise FileNotFoundError(f"Video file not created at expected location: {output_video}")

    except subprocess.CalledProcessError as e:
        print("❌ Error generating video:", e)
        print("Command output:", e.output if hasattr(e, 'output') else 'No output')
        print("Command stderr:", e.stderr if hasattr(e, 'stderr') else 'No stderr')
        raise Exception(str(e))
    except Exception as e:
        print("❌ Error:", str(e))
        print("Exception type:", type(e))
        print("Exception args:", e.args)
        raise
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Merge video with correct audio using FFmpeg
def merge_audio(video_file, audio_file, final_output):
    try:
        ffmpeg_path = get_ffmpeg_path() 
        
        # Convert all paths to absolute paths
        video_file = os.path.abspath(video_file)
        audio_file = os.path.abspath(audio_file)
        final_output = os.path.abspath(final_output)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(final_output), exist_ok=True)

        print(f"Using FFmpeg from: {ffmpeg_path}")
        print(f"Video file: {video_file}")
        print(f"Audio file: {audio_file}")
        print(f"Final output: {final_output}")
        
        # Check if input files exist
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        subprocess.run([
            ffmpeg_path,
            "-i", video_file,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac",
            final_output,
            "-y"
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print("✅ Final video saved at:", final_output)
    except subprocess.CalledProcessError as e:
        print("❌ Error merging audio:", e)
        print("Command stderr:", e.stderr.decode() if hasattr(e, 'stderr') else 'No stderr')
        raise Exception(f"FFmpeg error: {str(e)}")
    except Exception as e:
        print("❌ Error:", str(e))
        raise


def preprocess_audio(input_audio, output_audio):
    """Preprocess audio without duration limit"""
    try:
        ffmpeg_path = get_ffmpeg_path()
        command = [
            ffmpeg_path,
            '-i', input_audio,
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',      # Convert to mono
            output_audio,
            '-y'
        ]
        subprocess.run(command, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/fetch_trends', methods=['GET'])
def get_trends():
    country = request.args.get('country', 'in')
    category = request.args.get('category', 'general')
    trends = fetch_trending_topics(country, category)
    return jsonify(trends)

@app.route('/video_generation')
def video_generation():
    return render_template('video.html')

@app.route('/generate_video', methods=['POST'])
def create_video():
    if 'image' not in request.files:
        return jsonify({'error': 'Missing image file'}), 400

    image_file = request.files['image']
    
    # Check if audio file is provided or use generated audio
    audio_file = request.files.get('audio')
    use_generated_audio = False
    
    if not audio_file:
        # Use generated audio if available
        generated_audio_path = os.path.join('uploads', 'generated_audio.mp3')
        if os.path.exists(generated_audio_path):
            use_generated_audio = True
            print("Using generated audio file")
        else:
            return jsonify({'error': 'No audio file provided and no generated audio found'}), 400

    if image_file and allowed_file(image_file.filename):
        # Resolve absolute upload directory under the app root
        upload_dir = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
        os.makedirs(upload_dir, exist_ok=True)

        image_path = os.path.join(upload_dir, 'input_image.jpg')
        audio_path = os.path.join(upload_dir, 'input_audio.mp3')
        processed_audio = os.path.join(upload_dir, 'processed_audio.wav')

        # Save uploaded image
        image_file.save(image_path)
        
        # Save audio file or use generated audio
        if use_generated_audio:
            # Copy generated audio to input_audio.mp3
            import shutil
            shutil.copy2(generated_audio_path, audio_path)
            print("Copied generated audio to input_audio.mp3")
        else:
            # Save uploaded audio
            audio_file.save(audio_path)

        try:
            # Get audio duration before processing
            duration_cmd = [
                get_ffprobe_path(),
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            duration = float(subprocess.check_output(duration_cmd).decode().strip())
            print(f"Audio duration: {duration} seconds")

            # Preprocess audio without duration limit
            if not preprocess_audio(audio_path, processed_audio):
                return jsonify({'error': 'Audio preprocessing failed'}), 500

            # Generate video using processed audio
            generate_video(image_path, processed_audio, VIDEO_OUTPUT)
            merge_audio(VIDEO_OUTPUT, audio_path, FINAL_VIDEO)

            return jsonify({
                'success': True,
                'video_path': FINAL_VIDEO,
                'duration': duration
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    try:
        script = request.form.get('script')
        voice = request.form.get('voice') or 'en-US-JennyNeural'  # Use fallback if not provided

        if not script:
            return jsonify({'error': 'No script provided'}), 400

        # Ensure output directory exists
        os.makedirs('static/output', exist_ok=True)

        print(f"\n[generate_audio] Starting conversion...")
        print(f"Voice: {voice}")
        print(f"Script (first 100 chars): {script[:100]}")
        print(f"Saving to: {os.path.abspath(AUDIO_FILE)}")

        # Convert text to speech
        convert_text_to_speech(script, AUDIO_FILE, voice)

        # Confirm audio file was created
        if os.path.exists(AUDIO_FILE):
            print("✅ Audio file successfully created.")
            # Add a timestamp to the URL to prevent browser caching
            cache_buster = int(time.time())
            audio_path_with_version = f'/static/output/generated_audio.mp3?v={cache_buster}'
            return jsonify({
                'success': True,
                'audio_path': audio_path_with_version
            })
        else:
            print("❌ Audio file not found after TTS conversion.")
            return jsonify({'error': 'Audio file was not created successfully'}), 500

    except Exception as e:
        print(f"🔥 Exception in /generate_audio: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)