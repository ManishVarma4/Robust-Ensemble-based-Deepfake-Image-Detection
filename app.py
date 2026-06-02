import streamlit as st
from PIL import Image
import tempfile
import os
import io
import sys

# 👉 import your pipeline
from ensemble_decision import final_pipeline  

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🧠 Deepfake / Image Forgery Detector")
st.write("Upload an image to check if it's **Real or Fake**")

# Upload image
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    # Analyze button
    if st.button("🚀 Analyze Image"):

        with st.spinner("Processing..."):

            # 🔥 Capture all print output
            buffer = io.StringIO()
            sys.stdout = buffer

            try:
                result = final_pipeline(temp_path)
            except Exception as e:
                sys.stdout = sys.__stdout__
                st.error(f"❌ Error: {e}")
                st.stop()

            # Restore stdout
            sys.stdout = sys.__stdout__

            logs = buffer.getvalue()

        if result:
            st.success("✅ Analysis Complete")

            # 🎯 Final Result UI
            st.subheader("📊 Final Result")

            if result['result'] == "Real":
                st.success(f"🟢 REAL IMAGE ({result['confidence']:.2f}%)")
            else:
                st.error(f"🔴 FAKE IMAGE ({result['confidence']:.2f}%)")
                st.write(f"**Type:** {result.get('type', 'Unknown')}")

            # 🔍 Detailed logs
            with st.expander("🔍 Detailed Model Outputs"):
                st.code(logs)

        else:
            st.error("❌ Failed to process image")

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)