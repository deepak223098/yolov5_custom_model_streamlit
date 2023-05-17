import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# laod the custom trained weight into model
model = torch.hub.load("ultralytics/yolov5", 'custom',
                       path="last.pt", force_reload=True)

# add the title/slidebar
st.title("LOGO Detection of CD, DVD and PA with Streamlit")
st.slider("YOLOV5 Model")


@st.cache_data(show_spinner=False)
def read_image(img):
    return img


def yolov5_detect(img):
    results = model(img)
    if results is not None:
        # cv2.imshow("output", np.squeeze(results.render()))
        # cv2.waitKey(0)
        # results.print()
        ff = results.__repr__()
        print("ff", ff)
        st.subheader("Model OutPut:- " + ff[51:])
        st.image(results.render()[0].astype(np.uint8), use_column_width=True)


# img_type = st.selectbox("select image", ["Sample Image"])
# if img_type == "Sample Image":
#     image_url = r"data\images\val\disc1.png"
# else:
#     image_url = r"data\images\test\disc_tst_3.png"
#
# image = read_image(image_url)
# yolov5_detect(image)
upload_image = st.file_uploader("Choose an Image to detect", type="png")
print("upload_image", upload_image)
if upload_image is not None:
    img = Image.open(upload_image)
    img_array = np.array(img)
    im = read_image(img_array)
    yolov5_detect(im)
