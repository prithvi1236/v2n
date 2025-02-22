from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from deep_translator import GoogleTranslator
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import torch
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load AI models
T5_MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
summarization_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

def generate_heading(text):
    # Tokenize the text
    words = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Perform POS tagging
    tagged_words = nltk.pos_tag(words)

    # Select only nouns and adjectives (important keywords)
    important_words = [word for word, tag in tagged_words if tag.startswith('NN') or tag.startswith('JJ')]

    # Get the most common words
    word_counts = Counter(important_words)
    top_words = [word for word, _ in word_counts.most_common(3)]  # Adjust the number as needed

    # Generate heading
    heading = " ".join(top_words).title()
    return heading if heading else "No Keywords Found"
def fetch_transcript(youtube_url, supported_languages=["en", "hi", "es", "fr", "ml"], target_language="en"):
    """Fetches the transcript for a given YouTube video, translates if needed, and returns a timestamped dictionary."""
    try:
        video_id = youtube_url.split("v=")[-1]
        transcript_data = None
        detected_language = None

        for lang in supported_languages:
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                detected_language = lang
                break
            except:
                continue

        if not transcript_data:
            return None, "No transcript available."

        transcript_with_timestamps = {}
        for entry in transcript_data:
            start_time = entry["start"]
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            formatted_time = f"{minutes:02d}:{seconds:02d}"
            text = entry["text"]

            if detected_language != target_language:
                text = GoogleTranslator(source=detected_language, target=target_language).translate(text)
            
            transcript_with_timestamps[formatted_time] = text
        
        return transcript_with_timestamps, None
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except Exception as e:
        return None, f"Error fetching transcript: {e}"

def split_transcript(transcript_dict, max_words_per_segment=500):
    """Splits the transcript into segments of approximately max_words_per_segment words."""
    segments = []
    current_segment = []
    current_word_count = 0
    start_timestamp = None

    for timestamp, text in transcript_dict.items():
        words = text.split()
        
        if current_word_count + len(words) > max_words_per_segment:
            segments.append((start_timestamp, " ".join(current_segment)))
            current_segment = []
            current_word_count = 0
            start_timestamp = None

        if start_timestamp is None:
            start_timestamp = timestamp
        
        current_segment.extend(words)
        current_word_count += len(words)
    
    if current_segment:
        segments.append((start_timestamp, " ".join(current_segment)))
    
    return segments

def summarize_text(text):
    """Generates a summary for the given text using the T5 model."""
    try:
        input_text = "summarize: " + text.strip().replace("\n", " ")
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

        summary_ids = summarization_model.generate(
            input_ids,
            num_beams=4,
            length_penalty=2.0,
            max_length=150,
            min_length=40,
            early_stopping=True
        )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except:
        return "Summary unavailable"

def generate_title(text, max_length=15, min_length=5):
    """Generates a short and meaningful title for a text segment."""
    try:
        text = text.strip().replace("\n", " ")
        input_text = f"Generate a short, meaningful title: {text[:300]}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        title_ids = summarization_model.generate(
            input_ids, num_beams=6, temperature=0.7, top_p=0.9, length_penalty=1.5,
            max_length=max_length, min_length=min_length, early_stopping=True
        )
        return tokenizer.decode(title_ids[0], skip_special_tokens=True)
    except:
        return "Untitled Segment"

def convert_timestamp_to_seconds(timestamp):
    """Converts a timestamp string (MM:SS) into total seconds."""
    match = re.match(r"(\d+):(\d+)", timestamp)
    if match:
        minutes, seconds = map(int, match.groups())
        return minutes * 60 + seconds
    return None

@app.route("/process", methods=["POST"])
def process_youtube_video():
    """Processes a YouTube video link to generate segmented summaries with timestamps."""
    try:
        request_data = request.get_json()
        youtube_link = request_data.get("link")
        print("Received link:", youtube_link)
        if "v=" not in youtube_link:
            return jsonify({"error": "Invalid YouTube link."}), 400

        transcript, error = fetch_transcript(youtube_link)
        if error:
            return jsonify({"error": error}), 400

        segmented_transcripts = split_transcript(transcript)
        summarized_results = []

        for timestamp, segment_text in segmented_transcripts:
            segment_title = generate_heading(segment_text)
            segment_summary = summarize_text(segment_text)
            print(segment_summary)
            video_timestamp_link = youtube_link + "&t=" + str(convert_timestamp_to_seconds(timestamp)) + "s"
            summarized_results.append({
                "link": video_timestamp_link,
                "timestamp": timestamp,
                "title": segment_title,
                "summary": segment_summary
            })

        return jsonify({"segments": summarized_results})
    except Exception as e:
        return jsonify({"error": "Internal server error: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
