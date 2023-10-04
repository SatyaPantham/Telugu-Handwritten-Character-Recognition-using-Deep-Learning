import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas

def preprocessing(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def get_className(classNo):
    # Define your class name mappings here
    class_names = [
        "అ -A",
        "ఆ -AA",
        "ఇ -I",
        "ఈ -EE",
        "అఉ – U",
        "ఊ- UU",
        "ఋ – RU",
        "ౠ -ROO",
        "ఎ – E",
        "అః – AHAA",
        "ఐ – AI/AY",
        "ఒ – O",
        "ఓ – OO",
        "ఔ – AU",
        "అం – AUM",
        "ఏ – YE",
    ]
    if classNo >= 0 and classNo < len(class_names):
        return class_names[classNo]
    else:
        return "Character not recognized, please draw again"

# Load the trained model
model = load_model("model.h5")

# Streamlit UI
st.sidebar.image("./img/IBM2.png", use_column_width=True)
st.sidebar.title("Team Analytical Eyes")
st.sidebar.markdown("- Rahul Sharma\n- Satya Pantham \n- Kampasati Mahesh\n- Mohammad Mazid ")

st.title("Handwritten Telugu Character Recognition")
st.markdown("Draw a Telugu character below and click 'Predict'.")

# Create unique keys for the canvases
canvas_key_draw = "canvas_draw_" + str(id(st))
canvas_key_clear = "canvas_clear_" + str(id(st))

# Create a drawing canvas using st_canvas for drawing
canvas_result = st_canvas(
    stroke_width=3,
    stroke_color="black",
    background_color="#eee",
    height=300,  # Increased height for a larger canvas
    drawing_mode="freedraw",
    key=canvas_key_draw,  # Use a unique key
)

if st.button("Predict"):
    # Convert the canvas drawing to an image
    img = canvas_result.image_data.astype(np.uint8)

    # Save the drawn image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file_path = temp_file.name
        Image.fromarray(img).save(temp_file_path)

    # Load the saved image for processing
    drawn_image = cv2.imread(temp_file_path)

    # Process the image
    drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)
    drawn_image = cv2.resize(drawn_image, (50, 50))
    drawn_image = preprocessing(drawn_image)
    drawn_image = drawn_image.reshape(1, 50, 50, 1)

    # Make predictions
    prediction = model.predict(drawn_image)
    class_index = np.argmax(prediction, axis=1)

    # Get the class name
    result = get_className(class_index[0])

    # Display the result with a larger font
    st.subheader("Prediction:")
    st.write(result, font_size=24)  # Increase the font size

# Create a clear canvas button with a unique key
# if st.button("Delete", key=canvas_key_clear):
#     # Recreate the canvas with a new key to clear it
#     canvas_result = st_canvas(
#         stroke_width=3,
#         stroke_color="black",
#         background_color="#eee",
#         height=300,  # Increased height for a larger canvas
#         drawing_mode="freedraw",
#         key=canvas_key_draw,  # Use the same drawing canvas key
#     )
