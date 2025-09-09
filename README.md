# Audio Diarization

This Python script performs speaker diarization on audio files (e.g., MP3 or WAV). It uses OpenAI's Whisper for speech transcription, SpeechBrain for speaker embedding extraction, and KMeans clustering to assign speakers to transcription segments. The result is a formatted text file with transcribed text, timestamps, and speaker labels.

## Features
- Transcribes audio using Whisper's "turbo" model.
- Extracts speaker embeddings and clusters them to identify unique speakers.
- Supports automatic detection of speaker count or manual specification.
- Handles MP3 conversion to WAV via FFmpeg if needed.
- Outputs results to a TXT file.

## Requirements
- Python 3.8+
- Dependencies: `torch`, `numpy`, `sklearn`, `librosa`, `speechbrain`, `argparse`
- External tools: FFmpeg (for audio conversion), Whisper CLI (install via `pip install openai-whisper` or similar)

Install dependencies:
```
pip install torch numpy scikit-learn librosa speechbrain argparse
```
Ensure FFmpeg is installed (e.g., `brew install ffmpeg` on macOS or `apt install ffmpeg` on Ubuntu).

## Usage
Run the script from the command line:

```
python diarize.py <audio_file> [--speakers N] [-o output_file]
```

- `<audio_file>`: Path to the input audio file (required, e.g., `path/to/audio.mp3`).
- `--speakers N`: Expected number of speakers (optional, integer). If provided, the script uses this value for KMeans clustering (capped at the number of valid segments to avoid errors). If not provided, it automatically determines the optimal number using silhouette score analysis for better accuracy in unknown scenarios. Use this if you know the speaker count in advance to potentially improve clustering precision; otherwise, let it auto-detect for flexibility.
- `-o output_file` or `--output output_file`: Path to the output TXT file (optional). Defaults to the input filename with `.txt` extension in the current working directory (e.g., `audio.txt` for `audio.mp3`).

### Examples
- Basic run (auto-detect speakers, default output):
  ```
  python diarize.py path/to/audio.mp3
  ```

- Specify 2 speakers and custom output:
  ```
  python diarize.py path/to/audio.mp3 --speakers 2 -o results.txt
  ```

- Auto-detect speakers with custom output:
  ```
  python diarize.py path/to/audio.wav -o my_transcript.txt
  ```

## Output
The script generates a TXT file with:
- Summary: Number of detected speakers and total text length.
- Detailed segments: Each with speaker label, timestamps, and transcribed text.

Example output snippet:
```
================================================================================
TRANSCRIPTION RESULTS WITH SPEAKER DIARIZATION
================================================================================
Detected speakers: 2
Total text length: 1234 characters

Segment details:

[1] speaker_0 (0.0s-5.0s):
   Hello, this is speaker one.

[2] speaker_1 (5.0s-10.0s):
   Hi, responding as speaker two.
...
================================================================================
```

## Notes
- Logging is saved to `diarization.log` for debugging.
- The script handles errors gracefully and prints them to the console.
- For best results, ensure audio is clear and speakers are distinct.
- If clustering fails (e.g., too few segments), all segments default to "speaker_0".
