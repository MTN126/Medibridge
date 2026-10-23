# MediBridge

Scan a prescription, let a clinician verify it, then explain it to the patient in their own language.

> Hackathon prototype. Not a diagnostic tool or medical device. Use synthetic documents only.

## What it does

1. Upload a prescription or report image.
2. Gemma extracts medicines, allergies, diagnoses and abnormal labs. Unclear readings are flagged.
3. A clinician reviews and edits the data. Rule-based checks flag allergy matches and duplicate medicines.
4. After the clinician approves, Gemma explains it in Urdu, Roman Urdu, Punjabi or simple English, with optional audio.

## Model

Gemma 4 via the Gemini API, or locally via Ollama (`gemma4:e2b`).
