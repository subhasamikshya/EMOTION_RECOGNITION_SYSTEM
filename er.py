import cv2
from deepface import DeepFace
import sounddevice as sd
import numpy as np
import librosa
import soundfile as sf  # Added for saving audio files
import threading
import time

# Function to detect emotions from a video file or live feed
def detect_emotion_from_video(video_path=None):
    if video_path:
        cap = cv2.VideoCapture(video_path)  # Open the provided video file
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get the total number of frames and frame rate
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {fps}, Total Frames: {total_frames}")

        # Duration of the video in seconds
        video_duration = total_frames / fps
        print(f"Video Duration: {video_duration:.2f} seconds")
    else:
        cap = cv2.VideoCapture(0)  # Open the default camera
        video_duration = None  # Duration is not applicable for live feed

    start_time = time.time()  # Record the start time
    last_time_checked = start_time  # Time of the last check
    frame_count = 0  # Track frames processed

    while True:  # Continuous loop until we break manually
        current_time = time.time()
        
        # Check if we've reached the end of the video based on duration
        if video_path:
            elapsed_time = current_time - start_time
            if elapsed_time >= video_duration:
                print("Processed all frames within the video duration. Exiting...")
                break

        # Check if it's time to process the next frame (every 1 second)
        if current_time - last_time_checked >= 1:  # Process every 1 second
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame. Exiting...")
                break  # Exit the loop if no frame is captured or the video ends

            # Analyze emotion using DeepFace
            try:
                print("Frame captured successfully.")
                
                emotion_analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = emotion_analysis[0]['dominant_emotion']
                
                # Calculate elapsed time and display the detected emotion on the frame
                elapsed_time = current_time - start_time
                print(f"Detected Emotion at {elapsed_time:.2f} seconds: {dominant_emotion}")
                cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error in emotion detection: {e}")

            cv2.imshow('Emotion Recognition', frame)

            last_time_checked = current_time  # Update the last time checked
            frame_count += 1  # Increment the frame count

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Small wait to allow for key press
            print("Quitting emotion detection...")
            break

    cap.release()  # Release the video/camera capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Function to detect emotions from audio
def detect_emotion_from_audio(duration=5, sr=22050):
    print("Recording audio...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()  # Wait for the recording to finish
    print("Recording finished.")

    # Save the recorded audio temporarily using soundfile
    sf.write("temp_audio.wav", audio_data, sr)

    # Load the audio for analysis
    audio, _ = librosa.load("temp_audio.wav", sr=sr)

    # Here you can add emotion detection logic for audio
    # For now, we'll just print the shape of the audio data
    print("Audio data shape:", audio.shape)
    # Add more logic to analyze the audio and detect emotion.

# Main function to ask the user for input
def main():
    print("Choose an option:")
    print("1. Use live camera for emotion detection")
    print("2. Upload a video file for emotion detection")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        print("Starting live camera emotion detection...")
        video_thread = threading.Thread(target=detect_emotion_from_video)
        video_thread.start()
        video_thread.join()  # Wait for video processing to finish
    elif choice == '2':
        video_path = input("Enter the path to the video file: ")
        video_path = r"{}".format(video_path)  # Ensure the path is treated as a raw string
        print(f"Starting emotion detection from video file: {video_path}")
        video_thread = threading.Thread(target=detect_emotion_from_video, args=(video_path,))
        video_thread.start()
        video_thread.join()  # Wait for video processing to finish
    else:
        print("Invalid choice. Exiting...")

    # Optionally, you can also perform audio emotion detection AFTER video processing is done
    detect_emotion_from_audio()

if __name__ == '__main__':
    main()
