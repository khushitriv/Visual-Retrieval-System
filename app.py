
from pathlib import Path

import streamlit as st
from PIL import Image

from src.retrieval.retriever import ImageRetriever


PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Interior Image Search",
    page_icon="🏡",
    layout="wide",
)

st.title("Interior Image Search")
st.write(
    "Upload a room photo and discover visually similar interiors "
    "using CLIP, ResNet-50, or RGB color histograms."
)


@st.cache_resource
def get_retriever():
    return ImageRetriever()


retriever = get_retriever()

uploaded_file = st.file_uploader(
    "Upload an interior image",
    type=["jpg", "jpeg", "png", "webp"],
)

method = st.selectbox(
    "Retrieval method",
    ["CLIP", "ResNet-50", "Color Histogram"],
)

top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)

method_map = {
    "CLIP": "clip",
    "ResNet-50": "resnet",
    "Color Histogram": "color_histogram",
}

if uploaded_file is not None:
    temp_dir = PROJECT_ROOT / "outputs" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    query_path = temp_dir / f"query{suffix}"
    query_path.write_bytes(uploaded_file.getvalue())

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Your image")
        st.image(Image.open(uploaded_file), use_container_width=True)

    with st.spinner(f"Searching with {method}..."):
        try:
            results = retriever.retrieve(
                image_path=query_path,
                method=method_map[method],
                top_k=top_k,
            )
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.stop()

    with right:
        st.subheader(f"Top {len(results)} matches")
        for start in range(0, len(results), 2):
            cols = st.columns(2)

            for col, (_, result) in zip(
                cols,
                results.iloc[start:start + 2].iterrows(),
            ):
                image_path = Path(result["path"])
                with col:
                    if image_path.is_file():
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.warning(f"Image not found: {image_path}")

                    st.caption(
                        f"{result['room']} · {result['style']} "
                        f"· Score: {result['similarity']:.3f}"
                    )
else:
    st.info("Upload a room photo to start searching.")
