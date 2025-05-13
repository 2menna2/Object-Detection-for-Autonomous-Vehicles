import streamlit as st
import os
import time
from src.video_utils import run_inference_on_video

def main():
    st.set_page_config(page_title="YOLOv8 Vehicle Detection", page_icon="🚗", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🚗 Autonomous Vehicle Detection System</h1>", unsafe_allow_html=True)
    st.markdown("---")

    model_path = os.path.join("model", "best.onnx")
    if not os.path.exists(model_path):
        st.error("Model file not found. Please add 'best.onnx' to the model folder.")
        return

    st.header("Upload a Video File")
    uploaded_file = st.file_uploader("Choose a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if not uploaded_file:
        st.info("⏳ Please upload a video to begin processing.")
        return

    st.subheader("Original Video")
    st.video(uploaded_file)

    if st.button("Run Detection"):
        with st.spinner("Processing video... This may take some time."):
            os.makedirs("temp", exist_ok=True)
            temp_path = os.path.join("temp", uploaded_file.name)
            output_path = os.path.join("temp", "output_" + uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                start_time = time.time()
                run_inference_on_video(temp_path, output_path, model_path)
                st.success(f"Video processed successfully in {time.time() - start_time:.2f} seconds.")
                st.subheader("Detection Results")
                st.video(output_path)

                with open(output_path, "rb") as f:
                    st.download_button("Download Processed Video", data=f, file_name="processed_" + uploaded_file.name, mime="video/mp4")

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

            finally:
                if os.path.exists(temp_path): os.remove(temp_path)
                if os.path.exists(output_path): os.remove(output_path)

if __name__ == "__main__":
    main()