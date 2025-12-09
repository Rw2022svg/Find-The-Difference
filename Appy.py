import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import zipfile
import os
import time

# -- SESSION STATE INITIALIZATION FOR SETTINGS/MODAL --
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "show_settings_modal" not in st.session_state:
    st.session_state["show_settings_modal"] = False

# Automatically show settings modal on start if no API key present
if st.session_state["api_key"] == "":
    st.session_state["show_settings_modal"] = True

def open_settings_modal():
    st.session_state["show_settings_modal"] = True

# -- SETTINGS MODAL (with fallback if st.modal isn't available) --
def render_settings_modal():
    # Only render the modal when requested
    if not st.session_state.get("show_settings_modal", False):
        return

    # Use st.modal when available; otherwise fall back to a container (no crash).
    modal_ctx = getattr(st, "modal", None)

    # The settings form body (shared between modal and fallback)
    def settings_form_body():
        # Use a named form to keep behavior consistent
        with st.form("settings_form"):
            key_input = st.text_input(
                "Google AI Studio API Key",
                value=st.session_state.get("api_key", ""),
                type="password",
                placeholder="Enter your Gemini / AI Studio API key"
            )
            save_pressed = st.form_submit_button("Save")
            cancel_pressed = st.form_submit_button("Cancel")

            if save_pressed:
                st.session_state["api_key"] = key_input.strip()
                st.session_state["show_settings_modal"] = False
                # Rerun so UI reflects saved key immediately
                st.experimental_rerun()
            if cancel_pressed:
                # If no API key exists after cancel, keep modal showing next run.
                if not st.session_state.get("api_key"):
                    st.session_state["show_settings_modal"] = True
                else:
                    st.session_state["show_settings_modal"] = False
                st.experimental_rerun()

    if modal_ctx:
        # Newer Streamlit: show a real modal
        with modal_ctx("Settings — Gemini API Key"):
            settings_form_body()
    else:
        # Older Streamlit: fallback to container so app doesn't crash.
        # We render the same form inline with a small notice that it's not a modal.
        with st.container():
            st.info("Settings (modal not supported in this Streamlit version — using inline fallback)")
            st.header("Settings — Gemini API Key")
            settings_form_body()

# -- HELPER FUNCTION: GEMINI GENERATION ---
def generate_difference_pair_gemini(client, subject, style_prompt, diff_prompt):
    """
    Generates a side-by-side image using Gemini 2.5 Flash Image.
    NOTE: request the IMAGE modality (response_modalities=["IMAGE"]) instead of setting response_mime_type.
    """
    full_prompt = (
        f"Generate a single wide image split into two side-by-side panels. "
        f"Style: {style_prompt}. "
        f"Subject: {subject}. "
        f"The left panel is the base image. "
        f"The right panel is an identical copy but with exactly 3 VERY HARD TO SPOT differences. "
        f"Apply these specific difference types to the right panel: {diff_prompt}. "
        f"Do not add text, labels, or borders between panels. High resolution."
    )

    try:
        # Request IMAGE modality (remove response_mime_type which the SDK may treat as a text-only mime)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            # pass the prompt as a TextInput inside a list
            contents=[types.TextInput(text=full_prompt)],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                candidate_count=1
            )
        )

        # Navigate the response to find inline image bytes (keeps your existing parsing)
        # Different SDK versions may structure response slightly differently; this matches earlier logic.
        if getattr(response, "candidates", None):
            candidate = response.candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if parts:
                for part in parts:
                    if getattr(part, "inline_data", None):
                        return part.inline_data.data  # raw bytes

        st.warning("No image data found in response.")
        return None

    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# -- HELPER FUNCTION: PROCESSING ---
def process_and_save_images(image_bytes, index, subject, output_folder):
    """
    Splits the side-by-side image bytes and saves as pair.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")

        width, height = img.size
        half_width = width // 2

        img_base = img.crop((0, 0, half_width, height)).convert("RGB")
        img_diff = img.crop((half_width, 0, width, height)).convert("RGB")

        safe_subject = subject.replace("/", "_").strip() or "subject"
        base_name = f"{safe_subject} {index}.png"
        diff_name = f"{safe_subject} {index} (1).png"

        base_path = os.path.join(output_folder, base_name)
        diff_path = os.path.join(output_folder, diff_name)

        img_base.save(base_path)
        img_diff.save(diff_path)

        return [base_path, diff_path]
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# -- STREAMLIT UI --
st.set_page_config(page_title="Gemini 2.5 Diff Generator", layout="wide")

# Sidebar: settings button + info
with st.sidebar:
    st.header("Settings")
    if st.button("Change API Key"):
        open_settings_modal()
    if st.session_state.get("api_key"):
        # Masked display (show only last 4 chars)
        k = st.session_state["api_key"]
        masked = "•" * max(0, len(k) - 4) + (k[-4:] if len(k) >= 4 else k)
        st.write("Current key:", masked)
    else:
        st.write("No API key set")
    st.info("Get your key at https://aistudio.google.com/")

# Render modal if needed
render_settings_modal()

st.title("⚡ Gemini 2.5 Flash: Spot the Difference Generator")
st.markdown("Generates **hardcore** difference pairs using Google's cost-effective `gemini-2.5-flash-image` model.")

# Inputs
col1, col2 = st.columns([3, 1])
with col1:
    subject_input = st.text_input("Subject", placeholder="e.g. A busy robot factory floor")
with col2:
    num_pairs = st.number_input("Number of Pairs", min_value=1, max_value=50, value=1)

# The rigorous difference logic
diff_logic = """
1. Micro-Deletions: Remove tiny functional parts (screws, buttons) or one item in a cluster.
2. Material & Finish: Remove white shine/specular highlights on one object.
3. Geometric Nudges: Rotate or shift an object by pixels.
4. Pattern Disruptions: Shift texture stripes slightly or rotate internal circular details.
5. Physics Anomalies: Cast a shadow in the wrong direction for one object only.
"""
style_logic = "Old line cartoon style with bright flat colors."

# Use the API key stored in session_state
api_key = st.session_state.get("api_key", "").strip()

if st.button("Generate Images"):
    if not api_key:
        st.error("Please set your Google API Key in Settings.")
        # force open settings modal so user can enter it
        open_settings_modal()
        render_settings_modal()
    elif not subject_input:
        st.error("Please enter a subject.")
    else:
        # Initialize Google GenAI Client
        client = genai.Client(api_key=api_key)

        # specific folder for this run
        timestamp = int(time.time())
        temp_dir = f"temp_gen_gemini_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)

        st.write(f"Starting generation for: **{subject_input}**")
        progress_bar = st.progress(0)
        status_text = st.empty()

        generated_files_batch = []
        batch_counter = 0

        for i in range(1, int(num_pairs) + 1):
            status_text.text(f"Generating Pair {i}/{int(num_pairs)} using Gemini 2.5 Flash...")

            # Generate
            img_bytes = generate_difference_pair_gemini(client, subject_input, style_logic, diff_logic)

            if img_bytes:
                # Process and Split
                files = process_and_save_images(img_bytes, i, subject_input, temp_dir)
                if files:
                    generated_files_batch.extend(files)
                    try:
                        st.image(files[0], caption=f"Pair {i} Base (Gemini Generated)", width=300)
                    except Exception:
                        pass
                else:
                    st.warning(f"Failed to split pair {i}, skipping.")
            else:
                st.warning(f"Failed to generate pair {i} (Safety filter or Error), skipping.")

            # Update Progress
            progress_bar.progress(i / int(num_pairs))

            # Check for batch download (Every 10 or at the end)
            if i % 10 == 0 or i == int(num_pairs):
                if generated_files_batch:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zf:
                        for file_path in generated_files_batch:
                            zf.write(file_path, arcname=os.path.basename(file_path))

                    st.success(f"Batch Ready! (Pairs {batch_counter * 10 + 1} to {i})")
                    st.download_button(
                        label=f"⬇️ Download Batch {batch_counter + 1}",
                        data=zip_buffer.getvalue(),
                        file_name=f"{subject_input}_gemini_batch_{batch_counter + 1}.zip",
                        mime="application/zip"
                    )

                    # Clear batch list for next batch
                    generated_files_batch = []
                    batch_counter += 1

        st.success("Job Complete!")
