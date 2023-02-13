import numpy as np
import streamlit as st
import torch

from model import Generator


BATCH_SIZE = 128
NUM_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0

NOISE_DIM = 100
NUM_CLASSES = 2
EMBEDDING_DIM = BATCH_SIZE
FEATURES_G = 32


def generate_img_page():
    """
    Page to generate artificial images
    """

    st.title("Generate Artificial Image")

    datasets = (
        "Pneumonia Chest X-Ray",
        "Brain Tumour X-Ray",
    )

    chest_categories = ("Normal Chest X-Ray", "Pneumonia Chest X-Ray")
    brain_categories = ("Normal Brain X-Ray", "Tumour Brain X-Ray")
    # DNN_ModelNames=("LSTM", "BiLSTM", "BiLSTM+CNN")
    # Transformers=("mBERT","BanglaBERT","XLM-RoBERTa")

    approach = st.sidebar.selectbox("Approach", datasets)

    model_name = ""
    if approach == datasets[0]:
        model_name = st.selectbox("X-Ray Category", chest_categories)
    elif approach == datasets[1]:
        model_name = st.selectbox("X-Ray Category", brain_categories)

    generate = st.button("Generate")
    if generate:
        text = f"Choice: "

        if model_name == chest_categories[0]:
            text = f"Choice: {chest_categories[0]}"

            image_shape = (3, 128, 128)
            generator = Generator(
                NOISE_DIM, image_shape, NUM_CLASSES, EMBEDDING_DIM, FEATURES_G, NUM_GPU
            )
            generator.load_state_dict(torch.load('generator.pth'))
            generator.eval()

            x = torch.randn(2, NOISE_DIM, 1, 1)
            y = torch.zeros(2, dtype=torch.int64)

            with torch.no_grad():
                result = generator(x, y)
                result = result[0].permute(1, 2, 0) * 0.5 + 0.5
                result = np.uint8(result * 255)

                st.image(result, caption=text)

        elif model_name == chest_categories[1]:
            text = f"Choice: {chest_categories[0]}"

            image_shape = (3, 128, 128)
            generator = Generator(
                NOISE_DIM, image_shape, NUM_CLASSES, EMBEDDING_DIM, FEATURES_G, NUM_GPU
            )
            generator.load_state_dict(torch.load('generator.pth'))
            generator.eval()

            x = torch.randn(2, NOISE_DIM, 1, 1)
            y = torch.ones(2, dtype=torch.int64)

            with torch.no_grad():
                result = generator(x, y)
                result = result[0].permute(1, 2, 0) * 0.5 + 0.5
                result = np.uint8(result * 255)

                st.image(result, caption=text)

        elif model_name == brain_categories[0]:
            text = f"Choice: {brain_categories[0]}"

            image_shape = (1, 128, 128)
            generator = Generator(
                NOISE_DIM, image_shape, NUM_CLASSES, EMBEDDING_DIM, FEATURES_G, NUM_GPU
            )
            generator.load_state_dict(torch.load('tumor_generator.pt'))
            generator.eval()

            x = torch.randn(2, NOISE_DIM, 1, 1)
            y = torch.ones(2, dtype=torch.int64)

            with torch.no_grad():
                result = generator(x, y)
                result = result[0].permute(1, 2, 0) * 0.5 + 0.5
                result = np.uint8(result * 255)

                st.image(result, caption=text)

        elif model_name == brain_categories[1]:
            text = f"Choice: {brain_categories[1]}"

            image_shape = (1, 128, 128)
            generator = Generator(
                NOISE_DIM, image_shape, NUM_CLASSES, EMBEDDING_DIM, FEATURES_G, NUM_GPU
            )
            generator.load_state_dict(torch.load('tumor_generator.pt'))
            generator.eval()

            x = torch.randn(2, NOISE_DIM, 1, 1)
            y = torch.zeros(2, dtype=torch.int64)

            with torch.no_grad():
                result = generator(x, y)
                result = result[0].permute(1, 2, 0) * 0.5 + 0.5
                result = np.uint8(result * 255)

                st.image(result, caption=text)

        st.subheader(f"{text}")
