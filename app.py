# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Animal and Breed Detection",
    page_icon="🐈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Animal and Breed Detection")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Cat/Dog Breed Detection', 'Cat/Dog Skin Disease Detection','Birds Species Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Cat/Dog Breed Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Cat/Dog Skin Disease Detection':
    model_path = Path(settings.DETECTION_MODEL2)
if model_type == 'Birds Species Detection':
    model_path = Path(settings.DETECTION_MODEL3)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                names = res[0].names
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                class_detections_values = []
                for k, v in names.items():
                    class_detections_values.append(res[0].boxes.cls.tolist().count(k))
                # create dictionary of objects detected per class
                classes_detected = dict(zip(names.values(), class_detections_values))
                # st.text(classes_detected)
                for breed, detected in classes_detected.items():
                    if 'cat-' in breed and detected > 0:
                        cat_breed = breed.split('-')[1]
                        st.info(f"#### Animal: :blue[Cat] : {detected}")
                        st.success(f"####  Breed: :blue[{cat_breed}]")
                    if 'dog-' in breed and detected > 0:
                        cat_breed = breed.split('-')[1]
                        st.info(f"#### Animal: :blue[Dog : {detected}]")
                        st.success(f"####  Breed: :blue[{cat_breed}]")
                    if 'acorn-woodpecker' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'acorn-woodpecker'}]")
                    if 'annas-hummingbird' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'annas-hummingbird'}]")
                    if 'blue-jay' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'blue-jay'}]")
                    if 'blue-winged-warbler' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'blue-winged-warbler'}]")
                    if 'carolina-chickadee' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'carolina-chickadee'}]")
                    if 'carolina-wren' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'carolina-wren'}]")
                    if 'chipping-sparrow' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'chipping-sparrow'}]")
                    if 'common-eider' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'common-eider'}]")
                    if 'common-yellowthroat' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'common-yellowthroat'}]")
                    if 'dark-eyed-junco' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'dark-eyed-junco'}]")
                    if 'eastern-bluebird' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'eastern-bluebird'}]")
                    if 'eastern-towhee' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'eastern-towhee'}]")
                    if 'harris-hawk' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'harris-hawk'}]")
                    if 'hermit-thrush' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'hermit-thrush'}]")
                    if 'indigo-bunting' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'indigo-bunting'}]")
                    if 'juniper-titmouse' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'juniper-titmouse'}]")
                    if 'northern-cardinal' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'northern-cardinal'}]")
                    if 'northern-mockingbird' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'northern-mockingbird'}]")
                    if 'northern-waterthrush' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'northern-waterthrush'}]")
                    if 'orchard-oriole' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'orchard-oriole'}]")
                    if 'painted-bunting' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'painted-bunting'}]")
                    if 'prothonotary-warbler' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'prothonotary-warbler'}]")
                    if 'red-winged-blackbird' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'red-winged-blackbird'}]")
                    if 'rock-pigeon' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'rock-pigeon'}]")
                    if 'rofous-crowned-sparrow' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'rofous-crowned-sparrow'}]")
                    if 'rock-pigeon' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'rock-pigeon'}]")
                    if 'ruddy-duck' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'ruddy-duck'}]")
                    if 'scarlet-tanager' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'scarlet-tanager'}]")
                    if 'snow-goose' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'snow-goose'}]")
                    if 'song-sparrow' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'song-sparrow'}]")
                    if 'tufted-titmouse' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'tufted-titmouse'}]")
                    if 'varied-thrush' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'varied-thrush'}]")
                    if 'white-breasted-nuthatch' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'white-breasted-nuthatch'}]")
                    if 'white-throated-sparrow' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'white-throated-sparrow'}]")
                    if 'wood-duck' in breed and detected > 0:
                        st.info(f"#### Animal: :blue[Bird] : {detected}")
                        st.success(f"####  Species: :blue[{'wood-duck'}]")
                    if 'bacterial-dermatosis' in breed and detected > 0:
                        st.info(f"#### Disease: :blue[bacterial-dermatosis]")
                    if 'fungal-infection' in breed and detected > 0:
                        st.info(f"#### Disease: :blue[fungal-infection]")
                    if 'healthy' in breed and detected > 0:
                        st.info(f"#### Disease: :blue[healthy]")
                    if 'hypersensitivity-allergic-dermatosis' in breed and detected > 0:
                        st.info(f"#### Disease: :blue[hypersensitivity-allergic-dermatosis]")

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
