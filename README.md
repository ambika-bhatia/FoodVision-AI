# FoodVision-AI 

An image classification app that identifies food items from photos using deep learning. Built with TensorFlow and deployed as an interactive Streamlit app.

## What it does

Upload a photo of food, and the model predicts what dish it is — along with a confidence score. Trained to recognize **10 food categories**:

`apple_pie` · `bibimbap` · `cannoli` · `edamame` · `falafel` · `french_toast` · `ice_cream` · `ramen` · `sushi` · `tiramisu`

## Model

Started with a CNN trained from scratch (~29% accuracy), then switched to transfer learning using **MobileNetV2** — improved accuracy to **~83%**.

## Running the app

1. Clone the repo and set up a virtual environment
2. Install dependencies:
   ```
   pip install tensorflow streamlit pillow numpy
   ```
3. Run the app:
   ```
   cd app
   streamlit run app.py
   ```
4. Upload a food image (or use one from `sample_images/`) and see the prediction

## Tech Stack

- **TensorFlow / Keras** — model building and training
- **MobileNetV2** — pretrained base for transfer learning
- **Streamlit** — web app interface
- **Python**
