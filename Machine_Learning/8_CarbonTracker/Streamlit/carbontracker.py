import os
import plotly.io as pio
import streamlit as st
from PIL import Image

# Dossier où se trouve ce script (carbontracker.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.markdown(
    "<h1 style='color: mediumorchid;'>Production of CO2 in Deep Learning Models </h1>",
    unsafe_allow_html=True
)

st.header("1. Library CarbonTracker")

st.subheader("Publication of the library CarbonTracker")

st.write("CarbonTracker is a Python package that tracks and predicts the carbon footprint of training deep learning models. It estimates the energy" 
         "consumption and CO2 emissions associated with training deep learning models, helping researchers and practitioners make more " 
         "informed decisions about their model training processes.")
# On construit le chemin vers l'image dans le même dossier
img_publi = os.path.join(BASE_DIR, "Publication.jpg")

image = Image.open(img_publi)
st.image(image, caption="Publication of Lasse et al. 2020 https://www.researchgate.net/publication/342763375_Carbontracker_Tracking_and_Predicting_the_Carbon_Footprint_of_Training_Deep_Learning_Models", use_container_width=True)
st.markdown("[Lien vers le site CarbonTracker](https://docs.carbontracker.info/)", unsafe_allow_html=True)   


st.subheader("Methods to estimate CO2 emissions in Deep Learning")

st.write("CarbonTracker uses several methods to estimate CO2 emissions during the training of deep learning models:"
            "\n\n1. **Energy Consumption Measurement**: CarbonTracker measures the energy consumption of the hardware used for training, including CPUs and GPUs. It uses libraries like `psutil` to monitor power usage in real-time."
            "\n\n2. **Carbon Intensity Data**: The tool incorporates carbon intensity data from various sources, such as electricity grids, to estimate the CO2 emissions based on the energy consumed. This data can vary by location and time of day."
            "\n\n3. **Model Training Tracking**: CarbonTracker tracks the training process of deep learning models, recording metrics such as training time, hardware utilization, and energy consumption at each epoch."
            "\n\n4. **Prediction Models**: The tool includes predictive models that estimate future CO2 emissions based on historical training data and current training configurations."
            "\n\n5. **Reporting and Visualization**: CarbonTracker provides detailed reports and visualizations of the carbon footprint associated with model training, allowing users to analyze and optimize their training processes for lower emissions."
            )

st.subheader("For my project")

st.write("Before starting my project, I needed to understand how to measure and estimate the carbon footprint of my deep learning models."
         "I had to explore the characteristics of my hardware and the energy consumption patterns during model training."
        "\n\n1. **What hardware am I using? (CPU and GPU)**:"
        "\n\n My hardware => CPU: Intel(R) Core(TM) Ultra 7 155H  GPU: Intel® Arc™ graphics"
        "\n\n2. ** What are the maximum power consumption values for my CPU and GPU?**"
        "\n\n CPU TDP (Thermal Design Power): 115 W  GPU TGP (Total Graphics Power): 28 W"
        "\n\n3. ** What are the typical utilization rates of my CPU and GPU during training?**"
        "\n\n Based on research, it is possible to have exact values of CPU utilisation but for GPU, it is not the case."
        "\n\n I made assumptions for my project: I run a simple model with 50 epochs, so I assessed a CPU mean utilisation of 20%"
        "\n\n and GPU ? CPU and GPU utilization are not correlated ! Keras does not use GPU for simple models. So I assumed a GPU utilisation of 10% (plausible scenario)."
            ) 
            
code = '''tracker = CarbonTracker(epochs=max_epochs,verbose=1,
                        log_dir="./my_log_directory/",
                        sim_cpu="Intel(R) Core(TM) Ultra 7 155H",
                        sim_cpu_tdp=115,
                        sim_cpu_util=0.20,
                        sim_gpu="Intel® Arc™ graphics",
                        sim_gpu_watts=28, # https://nanoreview.net/en/gpu/intel-arc-igpu-8-cores
                        sim_gpu_util=0.1
                        )")'''
st.code(code, language="python")

st.subheader("Output example")
code2 = '''CarbonTracker: Average carbon intensity during training was 44.18 gCO2eq/kWh. 
CarbonTracker: 
Actual consumption for 10 epoch(s):
	Time:	0:00:56
	Energy:	0.002450397526 kWh
	CO2eq:	0.108256320595 g
	This is equivalent to:
	0.001013635961 km travelled by car
CarbonTracker: Finished monitoring.)")'''
st.code(code2, language="python")

st.header("2. Models and results")

st.subheader("Models trained")

st.write("I trained 7 different models with varying complexities (from very simple to more complex architectures) on the MNIST dataset."
         "\n\n **Model1**: Very simple feedforward neural network with 1 hidden layer (32 neurons)."
         "\n\n **Model2**: Feedforward neural network with 2 hidden layers (128 and 64 neurons)."
         "\n\n **Model3**: Feedforward neural network with 3 hidden layers (256, 128, and 64 neurons)."
        "\n\n **Model4**: Convolutional Neural Network (CNN) with 1 convolutional layers followed by 1 fully connected layers."
        "\n\n **Model5**: Deeper CNN with 2 convolutional layers and 2 fully connected layers."
        "\n\n **Model6**: Deeper CNN with 2 convolutional layers and 2 fully connected layers with dropout layers to prevent overfitting."
        "\n\n **Model7**: Deeper CNN with 2 convolutional layers and 2 fully connected layers with batchnormalization and globalaveragepooling.")


st.subheader("Results on accuracy and CO2 emissions")

img_val_acc_mod = os.path.join(BASE_DIR, "val_accuracy_par_modele.png")

image2 = Image.open(img_val_acc_mod)
st.image(image2, caption="Accuracy values by epochs for each models", use_container_width=True)
st.write("From model 1 to 3 => low accuracy."
            "\n\n From model 4 => significant increase in accuracy, reaching up to 92% for the most complex models."
            "\n\n Warning: Model 7 varying accuracy due to batchnormalization layers that add some randomness during training.")


st.subheader("Results CO2 emissions by models")

img_co2tot = os.path.join(BASE_DIR, "CO2Total_model.png")

image3 = Image.open(img_co2tot)
st.image(image3, caption="", use_container_width=True)   
            
st.write("As the model complexity increases from Model 1 to Model 7, there is a significant increase in CO2 emissions."
         "\n\n Model 1 has the lowest CO2 emissions, while Model 7 has the highest."
        "\n\n This trend indicates that more complex models require more computational resources, leading to higher energy consumption and CO2 emissions."
        "\n\n It highlights the trade-off between model performance and environmental impact, emphasizing the need for efficient model design and training practices to minimize carbon footprint in deep learning.")


st.subheader("Results CO2 equivalent to travel car by models")

img_co2eq = os.path.join(BASE_DIR, "CO2equivalenttrajetModel.png")

image4 = Image.open(img_co2eq)
st.image(image4, caption="", use_container_width=True)   
st.write("The CO2 emissions from training the models can be translated into equivalent distances traveled by a car."
         "\n\n Model 1 has the lowest equivalent distance, while Model 7 has the highest."
        "\n\n This comparison provides a tangible perspective on the environmental impact of training deep learning models, making it easier to understand and communicate the significance of CO2 emissions in terms of everyday activities like driving."
        "\n\n It underscores the importance of considering environmental sustainability in the development and deployment of machine learning models.")


st.subheader("Results Accuracy vs time Vss CO2 emissions")
img_acctime = os.path.join(BASE_DIR, "accuravyVStime.png")
image5 = Image.open(img_acctime)
st.image(image5, caption="", use_container_width=True)   

img_co2time = os.path.join(BASE_DIR, "co2VStime.png")
image6 = Image.open(img_co2time)
st.image(image6, caption="", use_container_width=True)   

st.write("These plots illustrate the relationship between model accuracy, training time, and CO2 emissions for different deep learning models.")
         