"""
MediBridge - scan a prescription / discharge summary, let a clinician verify it,
then explain it to the patient in their own language. Runs locally with Gemma via Ollama.

Run:  streamlit run app.py
"""
import base64
import io
import json
import re

import pandas as pd
import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/chat"

# language label -> (name used in prompt, gTTS code or None if no audio)
LANGS = {
    "Urdu (اردو)": ("Urdu, written in Urdu script", "ur"),
    "Roman Urdu": ("Roman Urdu (Urdu written in English letters)", None),
    "Punjabi (Shahmukhi)": ("Punjabi, written in Shahmukhi script", None),
    "English (simple)": ("very simple English", "en"),
}

EXTRACT_PROMPT = """You are a careful medical document reader. Read the attached document image.
Return ONLY valid JSON with exactly this schema:
{
  "patient_name": "string or empty",
  "diagnoses": ["string"],
  "allergies": ["string"],
  "medications": [
    {"name": "string", "dose": "string", "frequency": "string", "duration": "string",
     "confidence": "high or low"}
  ],
  "abnormal_labs": [
    {"test": "string", "value": "string", "note": "why it is abnormal if stated"}
  ],
  "uncertain_notes": ["anything you could not read clearly"]
}
Rules: never guess. If handwriting or a value is unclear, set confidence to "low" and
describe it in uncertain_notes. Do not invent information that is not in the document."""

EXPLAIN_PROMPT = """You are helping a patient understand their doctor-approved medical information.
Write in {lang}. Use very short, simple sentences a person with little education can follow.
For EACH medicine say: what it is generally used for, how much to take, and when to take it.
Then add a section "Go to the hospital immediately if" with warning signs relevant to the
conditions/medicines below, plus general emergency signs (trouble breathing, chest pain,
fainting, severe allergic reaction).
Do NOT diagnose anything new, do NOT change doses, and end with a line telling the patient
to ask their doctor or pharmacist if anything is unclear.

Approved information (JSON):
{data}
"""


def call_gemma(model: str, prompt: str, image_bytes: bytes | None = None, json_mode=False) -> str:
    msg = {"role": "user", "content": prompt}
    if image_bytes:
        msg["images"] = [base64.b64encode(image_bytes).decode()]
    payload = {
        "model": model,
        "messages": [msg],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def safety_checks(data: dict) -> list[str]:
    """Simple rule-based checks that do not depend on the model."""
    warnings = []
    allergies = [a.lower() for a in data.get("allergies", []) if a]
    names = []
    for med in data.get("medications", []):
        n = (med.get("name") or "").strip().lower()
        names.append(n)
        for a in allergies:
            if n and a and (a in n or n in a):
                warnings.append(f"ALLERGY ALERT: '{med.get('name')}' matches listed allergy '{a}'.")
        if str(med.get("confidence", "")).lower() == "low":
            warnings.append(f"Low confidence reading for '{med.get('name')}': verify against the original.")
    dupes = {n for n in names if n and names.count(n) > 1}
    for d in dupes:
        warnings.append(f"Duplicate medicine listed: '{d}'.")
    return warnings


def make_audio(text: str, code: str):
    try:
        from gtts import gTTS  # needs internet; fine for demo, mention offline TTS as future work

        buf = io.BytesIO()
        gTTS(text=text, lang=code).write_to_fp(buf)
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        st.warning(f"Audio unavailable: {e}")
        return None


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="MediBridge", page_icon="🩺", layout="wide")
st.title("🩺 MediBridge")
st.caption("Scan → Clinician verifies → Patient understands in their own language. "
           "Runs locally; no patient data leaves the machine. Decision support only, not a diagnosis.")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model name", value="gemma4:e2b",
                          help="Run `ollama list` to see the exact name you pulled.")
    lang_label = st.selectbox("Patient language", list(LANGS.keys()))
    st.info("Use synthetic / sample documents only. Never upload real patient data.")

left, right = st.columns(2)

with left:
    st.subheader("1. Upload document")
    up = st.file_uploader("Prescription / discharge summary (image)", type=["png", "jpg", "jpeg"])
    if up:
        st.image(up, use_container_width=True)
        if st.button("Extract with Gemma", type="primary"):
            with st.spinner("Gemma is reading the document..."):
                try:
                    raw = call_gemma(model, EXTRACT_PROMPT, up.getvalue(), json_mode=True)
                    st.session_state["data"] = parse_json(raw)
                    st.session_state.pop("explanation", None)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Extraction failed: {e}")

with right:
    st.subheader("2. Clinician review")
    data = st.session_state.get("data")
    if not data:
        st.write("Upload a document and click Extract.")
    else:
        for w in safety_checks(data):
            st.error(w) if w.startswith("ALLERGY") else st.warning(w)
        for n in data.get("uncertain_notes", []):
            st.warning(f"Unclear in document: {n}")

        data["patient_name"] = st.text_input("Patient name", data.get("patient_name", ""))
        data["diagnoses"] = [x.strip() for x in st.text_area(
            "Diagnoses (one per line)", "\n".join(data.get("diagnoses", []))).splitlines() if x.strip()]
        data["allergies"] = [x.strip() for x in st.text_area(
            "Allergies (one per line)", "\n".join(data.get("allergies", []))).splitlines() if x.strip()]

        st.markdown("**Medications**")
        meds = pd.DataFrame(data.get("medications", []),
                            columns=["name", "dose", "frequency", "duration", "confidence"])
        meds = st.data_editor(meds, num_rows="dynamic", use_container_width=True)
        data["medications"] = meds.fillna("").to_dict("records")

        st.markdown("**Abnormal labs**")
        labs = pd.DataFrame(data.get("abnormal_labs", []), columns=["test", "value", "note"])
        labs = st.data_editor(labs, num_rows="dynamic", use_container_width=True)
        data["abnormal_labs"] = labs.fillna("").to_dict("records")

        st.session_state["data"] = data
        approved = st.checkbox("✅ I have verified this against the original document")

        st.subheader("3. Patient explanation")
        if st.button("Generate patient explanation", disabled=not approved):
            lang_name, _ = LANGS[lang_label]
            patient_data = {k: v for k, v in data.items() if k != "uncertain_notes"}
            with st.spinner("Writing explanation..."):
                try:
                    st.session_state["explanation"] = call_gemma(
                        model, EXPLAIN_PROMPT.format(lang=lang_name,
                                                     data=json.dumps(patient_data, ensure_ascii=False)))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Generation failed: {e}")

        if "explanation" in st.session_state:
            text = st.session_state["explanation"]
            st.markdown(text)
            code = LANGS[lang_label][1]
            if code and st.button("🔊 Create audio"):
                audio = make_audio(re.sub(r"[#*_`]", "", text), code)
                if audio:
                    st.audio(audio, format="audio/mp3")
            elif not code:
                st.caption("Audio not available for this language in the demo.")