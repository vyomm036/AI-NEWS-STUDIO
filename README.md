# AI News Studio

AI News Studio is an end-to-end automated platform that transforms video content creation for news creators and social media influencers. By automating web scraping, sentiment analysis, script generation, audio synthesis, and lip-synced video production, it enables users to generate engaging videos from trending topics with minimal effort.

## Features

- **Automated Trending Topic Detection**: Fetches trending topics using NewsAPI.
- **Contextual News Collection**: Gathers relevant articles through DuckDuckGo.
- **Sentiment Analysis**: Uses VADER to analyze news sentiment and emotional tone.
- **AI Script Generation**: Produces natural language video scripts via Groq API (LLaMA model).
- **Text-to-Speech Conversion**: Transforms scripts to audio files using Edge TTS or similar tools.
- **Lip-Synced Video Creation**: Generates videos with facial lip-sync using Wav2Lip and uploaded images.
- **Project Management**: File explorer for HTML/CSS/JS file creation, download/export as ZIP.
- **Customizable Templates**: Offers template management for fast project creation.
- **Deployment Ready**: Supports both local and cloud deployments.

## Technologies Used

| Technology | Purpose |
|------------|---------|
| NewsAPI | Trending news detection |
| DuckDuckGo | Contextual news scraping |
| VADER | Sentiment analysis |
| Groq API, LLaMA | Script generation (natural language) |
| Text-to-Speech | Script-to-audio conversion |
| Wav2Lip | Lip-synced video generation |
| React.js, TypeScript, Tailwind CSS | Frontend development |
| Python (FastAPI/Flask) | Backend/API integration |

## System Architecture

- **Frontend**: Built on React.js, TypeScript, Tailwind CSS, featuring a Monaco editor, live preview, and project explorer.
- **Backend**: Python with FastAPI or Flask, managing AI integrations, file management, and data flow.
- **AI Model Integration Layer**: Uses LLaMA 3.3 70B (via Groq) for high-quality code generation and Gemma 3 1B LoRA for lightweight, fast generation.
- **Project Management Layer**: Supports multiple files, folders, and export/download options.

## How To Run

### 1. Frontend Setup

```bash
git clone <repo_url>
cd frontend
npm install
npm run dev
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# or for Flask:
python app.py
```

### 3. Model/API Configuration

- Insert your API keys for NewsAPI, DuckDuckGo, Groq, etc. into the config files.
- Make sure GPU support is available for running Wav2Lip and Gemma inference.

### 4. Usage

- Access the frontend (typically at `http://localhost:3000`).
- Select a trending topic and proceed through the script, audio, and video generation pipeline.

## Wav2Lip Integration

AI News Studio uses Wav2Lip for realistic lip-syncing in generated videos.

### Setup and Usage

Install Wav2Lip directly from GitHub:

```bash
pip install git+https://github.com/Rudrabha/Wav2Lip.git
```

Alternatively, clone the Wav2Lip repository manually:

```bash
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
```

Download the pretrained model weights (see the official Wav2Lip repository for the latest link) and place them in the `checkpoints/` directory.

Prepare your input files:

- `input_image.jpg` (static face image)
- `generated_audio.wav` (audio narration from the script)

Run Wav2Lip inference:

```bash
python inference.py --checkpoint_path checkpoints/wav2lip.pth \
  --face input_image.jpg --audio generated_audio.wav --outfile result_video.mp4
```

**Output:**
The generated `result_video.mp4` contains a realistic lip-synced video that can be used directly in your content pipeline.

For details, pretrained weights, and troubleshooting, visit the official [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip).

## Deployment

- **Frontend**: Can be deployed on Vercel or similar platforms.
- **Backend**: Suitable for Render, Railway, or local server deployment.
- Supports cloud-based and local GPU inference.

## Results & Performance

- Achieves high accuracy in prompt-to-code conversion (up to 93% for common components)[file:1].
- Real-time user experience with Monaco editor and live preview.
- Edge deployment support with lightweight models for fast local inference.

## References & Acknowledgements

- [Bolt.new AI-Powered Web Development][file:1]
- [Groq LLaMA Model Documentation][file:1]
- [Monaco Editor Integration][file:1]
- [Tailwind CSS Docs][file:1]
- [Wav2Lip GitHub Repository](https://github.com/Rudrabha/Wav2Lip)
