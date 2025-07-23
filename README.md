# FastSpeech2 on Hábrók — French Speech Synthesis

This project implements and customizes the open-source [FastSpeech2](https://github.com/ming024/FastSpeech2) model on the Hábrók cluster. It is part of the Speech Synthesis 2 Final Project in the Voice Technology MSc. 2024–2025.

## Original Repository

The base implementation was cloned from:
https://github.com/ming024/FastSpeech2

## Project Description

This project trains a FastSpeech2 model using a French-language dataset. Alignments were performed using Montreal Forced Aligner (MFA), and training was completed on GPU-enabled nodes of the Hábrók cluster. The resulting model synthesizes French speech from phoneme-level text input.

## What This Repository Contains

This folder only includes files that were modified or created during the project. The full codebase can be found in the original repository linked above.

Key modifications include:

- Alignment preprocessing for French using MFA
- Configuration updates for French phoneme-based synthesis
- Dataset restructuring to match expected input formats
- Scripts adapted for use on Hábrók infrastructure

## Language

This project was developed and trained using French as the target language.

## Directory Structure 

├── config/

│ └── [Updated YAML files for French dataset]

├── preprocessed_data/

│ └── [French-aligned TextGrid files]

├── scripts/

│ └── [Custom or modified training and synthesis scripts]

## How to Run

1. Clone the original FastSpeech2 repository
2. Install the required dependencies as listed in the original README
3. Place the French dataset in the expected structure
4. Run the preprocessing, training, and synthesis pipelines using the modified config and scripts



