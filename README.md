# SoruceX

SoruceX is a deep learning model for audio source separation. It is designed to take a mixed audio track and separate it into its constituent instrumental stems: drums, bass, vocals, and other instruments. The project includes scripts for training, evaluation, and inference, along with a simple Flask-based web interface for easy use.

## Features
*   Separates audio into four stems: drums, bass, vocals, and other.
*   Employs a multi-band U-Net architecture with a Transformer Encoder at its core.
*   Utilizes Rotary Positional Embeddings (RoPE) for improved temporal modeling.
*   Includes a simple web UI for uploading a `.wav` file and receiving a `.zip` file of the separated stems.
*   Provides complete scripts for training on the MUSDB dataset and evaluation using `museval`.

## Architecture
The core model (`AudioModel` in `SourceX.py`) processes the audio signal across four frequency bands to capture features at different resolutions:
1.  **Full Band:** The raw, unfiltered audio.
2.  **Low-Pass Band:** Below 400 Hz, processed with a Butterworth filter.
3.  **Band-Pass Band:** Between 400 Hz and 1900 Hz, processed with a Butterworth filter.
4.  **High-Pass Band:** Above 1900 Hz, processed with a Butterworth filter.

Each band is processed by a separate `BandModel`, which consists of:
*   An **Encoder:** A series of convolutional layers (`EncoderModule`) that downsample the input and create skip connections.
*   A **Transformer Encoder:** A Transformer block with Rotary Positional Embeddings (RoPE) at the bottleneck to process the latent representation.
*   A **Decoder:** A series of transposed convolutional layers (`DecoderModule`) that upsample the representation, incorporating the skip connections from the encoder.

The outputs from the four band models are concatenated and passed through a final Fully Connected (FC) layer to produce the separated stereo stems for each source.

## Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/runeen/SoruceX.git
    cd SoruceX
    ```
2.  Install the required Python packages. It is recommended to use a virtual environment.
    ```bash
    pip install torch torchaudio numpy scipy musdb museval tqdm flask torchinfo torchtune
    ```
3.  The model is configured to use a CUDA-enabled GPU. Ensure you have a compatible PyTorch version with CUDA support installed.

Note: There are no model checkpoints provided, you need to train the model yourself (by running the SourceX script)

### Web Interface (Easiest)
1.  Launch the Flask application:
    ```bash
    python flask_app.py
    ```
2.  Open your web browser and navigate to `http://127.0.0.1:5000`.
3.  Upload a `.wav` file using the form.
4.  A `.zip` archive containing the separated stems (`drums.wav`, `bass.wav`, `vocals.wav`, `other.wav`) will be downloaded automatically.

### Command-Line Inference
1.  Create an `input` directory and place your source audio file (e.g., `my_song.wav`) into it.
2.  Run the inference script:
    ```bash
    python inference.py
    ```
3.  The separated stems will be saved as `.wav` files in the `output/` directory.

### Evaluation
To evaluate the model's performance on the MUSDB test set:
1.  Ensure you have the `musdb` dataset installed and accessible.
2.  Run the evaluation script:
    ```bash
    python eval.py
    ```
3.  The script will iterate through the test tracks, perform separation, and print the `museval` scores (SDR, SIR, SAR, ISR) for each track and the aggregated results.

### Training
To train the model from scratch or continue a previous session:
1.  Ensure you have the `musdb` dataset installed and accessible.
2.  Run the training script:
    ```bash
    python SourceX.py
    ```
3.  The script will automatically handle data augmentation, batching, and training. Model checkpoints are saved to `istorie antrenari/azi/model.model` after processing batches, allowing for resumable training.

## Project Structure
```
.
├── SourceX.py          # Core model architecture and training loop
├── flask_app.py        # Flask web application for inference
├── inference.py        # Command-line script for separating a single file
├── eval.py             # Script for evaluating the model on the MUSDB dataset
├── augment.py          # Data augmentation functions for training
├── input/              # Directory for source audio files for inference
├── templates/
│   └── upload.html     # HTML template for the web UI
└── istorie antrenari/    # Directory containing training logs, notes, and model checkpoints
