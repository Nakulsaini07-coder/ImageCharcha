import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM,
    Dropout, Reshape, add, concatenate
)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

plt.rcParams['font.size'] = 12
sns.set_style("dark")
warnings.filterwarnings('ignore')


# Load tokenizer 
@st.cache_resource(show_spinner=False)
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        return tokenizer_from_json(f.read())

tokenizer = load_tokenizer()
vocab_size = len(tokenizer.word_index) + 1


# Load Densenet model
@st.cache_resource(show_spinner=False)
def load_densenet_model():
    base = DenseNet201()
    return Model(inputs=base.input, outputs=base.layers[-2].output)

densenet_model = load_densenet_model()

# custom web page title
st.set_page_config(page_title="Image Charcha")

# Streamlit app
st.title("Image Charcha")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained LSTM model."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

max_length = 34  

def build_model(vocab_size: int):
    input1 = Input(shape=(1920,), name="input_layer")
    input2 = Input(shape=(max_length,), name="input_layer_1")

    img_features = Dense(256, activation='relu', name="dense")(input1)
    img_features_reshaped = Reshape((1, 256), name="reshape", input_shape=(256,))(img_features)

    sentence_features = Embedding(vocab_size, 256, mask_zero=False, name="embedding")(input2)
    merged = concatenate([img_features_reshaped, sentence_features], axis=1, name="concatenate")
    sentence_features = LSTM(256, name="lstm")(merged)

    x = Dropout(0.5, name="dropout")(sentence_features)
    x = add([x, img_features], name="add")
    x = Dense(128, activation='relu', name="dense_2")(x)
    x = Dropout(0.5, name="dropout_1")(x)
    output = Dense(vocab_size, activation='softmax', name="dense_4")(x)

    caption_model = Model(inputs=[input1, input2], outputs=output)
    caption_model.compile(loss='categorical_crossentropy', optimizer='adam')
    return caption_model

def load_weights_manual(weights_path: str, model: Model):
    """Load weights saved as TF-trackable H5 (vars/0 style) into the compiled model."""
    with h5py.File(weights_path, "r") as f:
        deps = f["_layer_checkpoint_dependencies"]

        def read_vars(layer_name):
            group = deps[layer_name]["vars"]
            return [np.array(group[str(i)]) for i in sorted(map(int, group.keys()))]

        # Dense layers
        model.get_layer("dense").set_weights(read_vars("dense"))
        model.get_layer("dense_2").set_weights(read_vars("dense_2"))
        model.get_layer("dense_4").set_weights(read_vars("dense_4"))

        # Embedding
        model.get_layer("embedding").set_weights(read_vars("embedding"))

        # LSTM (kernel, recurrent_kernel, bias)
        lstm_vars = read_vars("lstm/cell")
        if len(lstm_vars) == 3:
            model.get_layer("lstm").set_weights(lstm_vars)
        else:
            raise ValueError("Unexpected number of LSTM weights; expected 3 tensors")

        # No weights for reshape, concat, add, dropout, inputs

@st.cache_resource(show_spinner=False)
def load_caption_model():
    weights_path = "model1.weights.h5"
    model = build_model(vocab_size)
    try:
        model.load_weights(weights_path)
    except ValueError as exc:
        try:
            load_weights_manual(weights_path, model)
        except Exception as inner_exc:
            st.error(
                "Failed to load model1.weights.h5 with both standard and manual loaders.\n"
                f"Standard loader error: {exc}\nManual loader error: {inner_exc}"
            )
            st.stop()
    return model

model = load_caption_model()

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image")

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Load image
        img = load_img(uploaded_image, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features
        features = densenet_model.predict(img_array, verbose=0)

        # Define function to get word from index
        def get_word_from_index(index, tokenizer):
            return next(
                (word for word, idx in tokenizer.word_index.items() if idx == index), None
        )

        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                caption += " " + predicted_word
                if predicted_word is None or predicted_word == "endseq":
                    break
            return caption
        
        # Generate caption
        generated_caption = predict_caption(model, features, tokenizer, max_length)

        # Remove startseq and endseq
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")

        # Display the generated caption with custom styling
        st.markdown(
            f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
            f'<p style="font-style: italic;">“{generated_caption}”</p>'
            f'</div>',
            unsafe_allow_html=True
        )
