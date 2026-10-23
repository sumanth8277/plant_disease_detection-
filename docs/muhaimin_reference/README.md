# 🌿 Plant Disease Classification and Recommendations

**Plant Disease Classification and Recommendations** is an AI-powered solution for identifying plant diseases and providing actionable recommendations. The project includes a deep learning model, **PlantCareNet (Custom CNN)**, trained on diverse plant disease datasets. Additionally, a **mobile app** is developed to allow users to upload images, detect diseases, and receive treatment recommendations.

---

## 🚀 Features  

### 📂 High-Quality & Diverse Dataset  
PlantCareNet is trained on a **carefully curated dataset** from multiple reputable sources, ensuring comprehensive coverage of plant diseases across various species.  

- 🌾 **PlantVillage** – A widely used benchmark dataset for plant disease classification.  
- 🌿 **New Bangladeshi Crop Disease Dataset** – Region-specific images covering common crop diseases in Bangladesh.  
- 🥦 **VegNet Cauliflower Dataset** – A specialized dataset for cauliflower disease identification.  
- 🍆 **Eggplant Disease Recognition Dataset** – Focused on detecting diseases affecting eggplants.  

The dataset underwent **extensive preprocessing and augmentation** to improve model generalization, making it robust for real-world applications.  


### 🌱 Deep Learning Model: PlantCareNet  
Our project is powered by **PlantCareNet**, a custom Convolutional Neural Network (CNN) specifically designed for plant disease classification. Unlike generic models, PlantCareNet is fine-tuned for **lightweight deployment**, making it suitable for mobile and edge applications. It has been trained on a diverse set of plant disease images to ensure high accuracy and robust generalization.  

- 🔹 **Custom CNN architecture** designed for plant disease identification.  
- 🔹 **Enhanced for mobile and real-time inference** with minimal latency.  
- 🔹 **Handles multiple plant species and disease categories with high precision.**  

---

### 📱 Mobile App for Disease Detection  
The project includes an **Android-based mobile application** that allows users to detect plant diseases simply by capturing an image. The app is designed with an intuitive interface, making it accessible to farmers, researchers, and agronomists.  

- 📷 **Easy-to-use interface** – Capture or upload images of the affected plant.  
- ⚡ **Real-time disease detection** – The AI model instantly processes the images.  
- 🌿 **Actionable recommendations** – Get disease information and treatment suggestions.
- 🔍 **Ref Mode & LLM Mode** – In Ref Mode, the app provides personalized recommendations based on a reference database tailored to the detected disease, offering actionable treatment suggestions. LLM Mode utilizes GPT to provide intelligent insights, answering user queries and suggesting appropriate actions, further enhancing the user experience with AI-driven support.---

---

## 📂 Repository Contents  
This repository includes the following:

- 📜 **Kaggle Notebook**: The Jupyter notebook used for training the **PlantCareNet** deep learning model. It includes data preprocessing, model architecture, training, and evaluation. The notebook is fully documented to allow easy reproduction of results.

- 📁 **Dataset**: The dataset used for training the model, which is a combination of multiple plant disease datasets such as **PlantVillage, New Bangladeshi Crop Disease, VegNet Cauliflower**, and **Eggplant Disease Recognition**. It contains images of various plant diseases across different crops like rice, corn, tomato, and more.

- 📱 **Base APK**: The compiled **Android app (.apk file)** for plant disease identification. This version of the app is ready for installation on Android devices and allows users to capture plant images, detect diseases, and get treatment recommendations.

- 📂 **Android Studio Project**: The complete source code of the mobile application developed using **Android Studio**. It contains all the necessary files for building the app from scratch, including the integration of the disease detection model, the user interface, and backend logic.

---

## 🛠 Setup and Installation

### **Prerequisites**
- **Kaggle Account**: Required to access and run the notebook. You can sign up [here](https://www.kaggle.com/).
- **Kaggle Notebook**: The model is trained in a **Kaggle Notebook** environment, which comes with pre-installed libraries such as **TensorFlow**, **Keras**, **NumPy**, and more.
- **Android Studio** (Optional): For modifying or building the Android app from source, download and install **Android Studio** from [here](https://developer.android.com/studio).

### **Running the Model in Kaggle**
1. Open the **Kaggle Notebook** provided in the repository.
2. Upload the dataset to the **Kaggle environment** (if needed).
3. Run the cells sequentially to preprocess the data, train the **PlantCareNet** model, and evaluate the results.
4. The trained model will be saved in the notebook for future inference.

### **Mobile App Setup**
1. **Base APK**: Download the **Base APK** from the repository and install it on your Android device for plant disease detection.
2. **Android Studio Project**: Open the **Android Studio Project** folder in Android Studio to modify or build the app from source.
   - After opening the project, build and deploy the app to an Android device or emulator.


### **Clone the Repository**
```sh
git clone https://github.com/github.com/Muhaimin008/Plant-disease-classification-and-recommendations.git
cd plant-disease-classification-and-recommendations
