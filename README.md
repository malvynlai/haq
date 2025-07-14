# 🧠 Pic2Plate – AI-Powered Culinary Companion

**Team LetsDOit | HaQathon 2025 – Qcom**

Pic2Plate is an AI-powered kitchen assistant that transforms the way users plan meals and manage groceries. By combining computer vision and natural language processing, the app identifies ingredients from a photo of your fridge and generates personalized recipes and shopping suggestions.

---

## What It Does

1. **Image Capture** 
 Snap a photo of your fridge (supports multiple angles or panoramic view).

2. **Ingredient Recognition** 
 Uses object detection and OCR to identify both packaged and unpackaged food items.

3. **Inventory Mapping** 
 Creates a virtual inventory of your fridge contents.

4. **Recipe Generation** 
 Matches available ingredients with a recipe database to generate meal ideas.

5. **Ingredient Enhancement** 
 Suggests additional ingredients to improve or complete recipes.

---

## 💡 Problem Statement

Many people struggle with:
- Deciding what to cook with what's already in their fridge.
- Wasting food due to forgotten or unused ingredients.
- Lack of time for meal planning or grocery list creation.

**Pic2Plate** addresses these issues by making AI accessible in everyday kitchen tasks.

---

## 🛠️ Tech Stack

- **Hardware:** Snapdragon X Elite
- **Libraries:** 
    - `onnxruntime-qnn` (for running the models)
    - `customtkinter`, `tkinterdnd2`, `tkinterweb`, `markdown2`, `tkinter` (for UI)
    - `opencv-python`, `pillow` (for image processing)
- **Models Used:**
    - Google Inception V3 (Image Recognition)
    - DeepSeek R1 Distill Qwen 1.5B (Language Model)
    - YOLO X (Object Detection)

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.13.3
- Snapdragon X Elite device with NPU support

### Downloading models
Create a folder called `models` and put the following in it:
- Download the [Inception v3 model](https://aihub.qualcomm.com/compute/models/inception_v3?domain=Computer+Vision&useCase=Image+Classification)
- Download the [Yolo-X model](https://aihub.qualcomm.com/compute/models/yolox?domain=Computer+Vision&useCase=Object+Detection)
- Download the `qnn-deepseek-r1-distill-qwen-1.5b.zip` file from [https://drive.google.com/drive/folders/1hCopYw7rMdeOm3zV6NC2do9orzpKqAMf](https://drive.google.com/drive/folders/1hCopYw7rMdeOm3zV6NC2do9orzpKqAMf) and unpack it into your `models` folder

### To avoid setting up the environment

We also provide a Windows executable for the ARM environment [here](https://drive.google.com/drive/folders/1odLu2Vyv1BrTvORCcg-MO3by8AnlE16x)

You still need to download the models and put the executable into this repository, 
but it lets you avoid setting up the `python` environments.

---

## Impact
- Saves time and money
- Improves cooking skills
- Promotes sustainability by reducing food waste
- Encourages smart shopping habits
- Reduces unnecessary trips and emissions

**Target Users:** 
Busy professionals, students, families, and health-conscious individuals.

---

## 👥 Team Members

- Sophia Curutchague – Marketing Intern 
- Nurullah Sevim – Engineering Intern 
- Akhil Bhimaraju – Engineering Intern 
- Malvyn Lai – Software Engineering Intern 
- Sai Madhav Modepalli – Engineering Intern 
- Akhil Reddy Peesari – Engineering Intern 

---