import streamlit as st
import traceback
from PIL import Image
from io import BytesIO
import zipfile
import os
import time

# Try to import Google GenAI SDK, give actionable guidance if fails
try:
    from google import genai
    from google.genai import types
except Exception as e:
    st.error(
        "Could not import the Google GenAI SDK (google.genai).\n\n"
        "If you haven't installed it, try one of these (depending on the package available):\n"
        "  pip install google-genai\n"
        "or\n"
        "  pip install google-generativeai\n\n"
        "Then restart Streamlit. Full import error:\n"
    )
    st.code(traceback.format_exc())
    st.stop()

# -- SESSION STATE INITIALIZATION FOR SETTINGS/MODAL/DEBUG --
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "show_settings_modal" not in st.session_state:
    st.session_state["show_settings_modal"] = False
if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = False

# Automatically show settings modal on start if no API key present
if st.session_state["api_key"] == "":
    st.session_state["show_settings_modal"] = True

def open_settings_modal():
    st.session_state["show_settings_modal"] = True

# -- SETTINGS MODAL (with fallback if st.modal isn't available) --
def render_settings_modal():
    if not st.session_state.get("show_settings_modal", False):
        return

    modal_ctx = getattr(st, "modal", None)

    def settings_form_body():
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
                st.experimental_rerun()
            if cancel_pressed:
                if not st.session_state.get("api_key"):
                    st.session_state["show_settings_modal"] = True
                else:
                    st.session_state["show_settings_modal"] = False
                st.experimental_rerun()

    if modal_ctx:
        with modal_ctx("Settings — Gemini API Key"):
            settings_form_body()
    else:
        with st.container():
            st.info("Settings (modal not supported in this Streamlit version — using inline fallback)")
            st.header("Settings — Gemini API Key")
            settings_form_body()

# -- HELPER FUNCTION: GEMINI GENERATION ---
def generate_difference_pair_gemini(client, subject, style_prompt, diff_prompt, debug=False):
    """
    Generates a side-by-side image using Gemini 2.5 Flash Image.
    Returns raw image bytes or None on failure.
    This function now gracefully handles SDK versions that don't expose types.TextInput.
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

    # Build contents in a version-tolerant way:
    used_typed_input = False
    try:
        # Preferred: typed TextInput (may not exist in some SDK builds)
        TextInput = getattr(types, "TextInput", None)
        if TextInput:
            text_input = TextInput(text=full_prompt)
            used_typed_input = True
        else:
            # If not present, fall back to a plain dict. Many SDK wrappers accept dicts.
            text_input = {"text": full_prompt}
    except Exception:
        # If constructing the typed input raises for some reason, fallback to dict
        text_input = {"text": full_prompt}

    if debug:
        st.write(f"Using typed TextInput: {bool(used_typed_input)}")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[text_input],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                candidate_count=1
            )
        )

        if debug:
            st.text("Raw SDK response (repr):")
            try:
                st.code(repr(response))
            except Exception:
                st.write("Unable to show full repr of response. See logs.")

        # Try to find inline image bytes in several known shapes of responses
        if getattr(response, "candidates", None):
            candidate = response.candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if parts:
                for part in parts:
                    if getattr(part, "inline_data", None):
                        return part.inline_data.data  # raw bytes

        # Defensive recursive search (covers unexpected SDK shapes)
        def find_inline_bytes(obj):
            if obj is None:
                return None
            if hasattr(obj, "inline_data") and getattr(obj.inline_data, "data", None):
                return obj.inline_data.data
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    res = find_inline_bytes(v)
                    if res:
                        return res
            if isinstance(obj, dict):
                for v in obj.values():
                    res = find_inline_bytes(v)
                    if res:
                        return res
            if hasattr(obj, "__dict__"):
                for v in vars(obj).values():
                    res = find_inline_bytes(v)
                    if res:
                        return res
            return None

        found = find_inline_bytes(response)
        if found:
            return found

        st.warning("No image data found in response. The request may have been blocked by safety filters or the SDK returned unexpected structure.")
        return None

    except Exception:
        st.error("Error generating image via GenAI SDK. Full traceback:")
        st.code(traceback.format_exc())
        return None

# -- HELPER FUNCTION: PROCESSING ---
def process_and_save_images(image_bytes, index, subject, output_folder):
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
    except Exception:
        st.error("Error processing image. Full traceback:")
        st.code(traceback.format_exc())
        return None

# -- STREAMLIT UI --
st.set_page_config(page_title="Gemini 2.5 Diff Generator", layout="wide")

# Sidebar: settings button + info + debug toggle
with st.sidebar:
    st.header("Settings")
    if st.button("Change API Key"):
        open_settings_modal()
    if st.session_state.get("api_key"):
        k = st.session_state["api_key"]
        masked = "•" * max(0, len(k) - 4) + (k[-4:] if len(k) >= 4 else k)
        st.write("Current key:", masked)
    else:
        st.write("No API key set")
    st.info("Get your key at https://aistudio.google.com/")
    st.checkbox("Enable debug mode (show raw responses & tracebacks)", value=st.session_state["debug_mode"], key="debug_mode")

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

diff_logic = """
1. Micro-Deletions: Remove tiny functional parts (screws, buttons) or one item in a cluster.
2. Material & Finish: Remove white shine/specular highlights on one object.
3. Geometric Nudges: Rotate or shift an object by pixels.
4. Pattern Disruptions: Shift texture stripes slightly or rotate internal circular details.
5. Physics Anomalies: Cast a shadow in the wrong direction for one object only.
"""
style_logic = "Old line cartoon style with bright flat colors."

api_key = st.session_state.get("api_key", "").strip()

# Diagnostic button: test API key + client creation (does not call heavy generate operations)
if st.button("Run client diagnostic (no generation)"):
    if not api_key:
        st.error("Please set your Google API Key in Settings.")
        open_settings_modal()
        render_settings_modal()
    else:
        try:
            client = genai.Client(api_key=api_key)
            st.success("Client created successfully.")
            if hasattr(client, "models"):
                st.write("Client has 'models' attribute.")
            else:
                st.warning("Client does not expose 'models'. SDK version may differ. Show client repr below:")
                st.code(repr(client))
        except Exception:
            st.error("Failed to create client. Full traceback:")
            st.code(traceback.format_exc())

if st.button("Generate Images"):
    if not api_key:
        st.error("Please set your Google API Key in Settings.")
        open_settings_modal()
        render_settings_modal()
    elif not subject_input:
        st.error("Please enter a subject.")
    else:
        client = None
        try:
            client = genai.Client(api_key=api_key)
        except Exception:
            st.error("Failed to create GenAI client. Full traceback:")
            st.code(traceback.format_exc())
            st.stop()

        timestamp = int(time.time())
        temp_dir = f"temp_gen_gemini_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)

        st.write(f"Starting generation for: **{subject_input}**")
        progress_bar = st.progress(0)
        status_text = st.empty()

        generated_files_batch = []
        batch_counter = 0
        debug = st.session_state.get("debug_mode", False)

        for i in range(1, int(num_pairs) + 1):
            status_text.text(f"Generating Pair {i}/{int(num_pairs)} using Gemini 2.5 Flash...")

            img_bytes = generate_difference_pair_gemini(client, subject_input, style_logic, diff_logic, debug=debug)

            if img_bytes:
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

            progress_bar.progress(i / int(num_pairs))

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

                    generated_files_batch = []
                    batch_counter += 1

        st.success("Job Complete!")
