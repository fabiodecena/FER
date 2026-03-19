# 🎯 Marketing Research Sentiment Probe (v1.0)

This application is a professional-grade facial sentiment analysis tool designed for **unbiased consumer research**. It utilizes a **ConvNeXt-Tiny** deep learning architecture to perform real-time emotion inference directly on the edge.

---

## 📖 User Instructions: How to Use the App

To ensure the highest quality data for our research, please follow these simple steps:

1.  **Preparation:** * Ensure you are in a quiet, well-lit environment. 
    * Avoid having bright lights or windows directly behind you (backlighting), as this makes it difficult for the AI to see your facial features.
2.  **Launch:** * Double-click the `inference_webcam.exe` file. 
    * A window titled "Marketing Study - Participant Feed" will appear.
3.  **The "Ghost Mirror":** * You will see a live video feed of yourself. This is a clean "mirror" provided for your comfort. 
    * **Note:** To prevent "Observer Bias," you will not see any AI bounding boxes or emotion labels on your screen. This allows you to react naturally to the media being tested.
4.  **Automatic Capture:** * The system is "hands-free." It will automatically detect and save snapshots of high-confidence emotional expressions. You do not need to click anything.
5.  **Termination:** * To end the session, simply press the **'Q'** key on your keyboard. The window will close and the data logs will be finalized.

---

## 🔐 Privacy Policy & Data Consent Agreement

By using this application, you acknowledge and agree to the following data processing terms:

### 1. Scope of Data Collection
This application captures two types of data:
* **Numerical Metadata:** Timestamps, predicted emotion categories (e.g., Happy, Neutral), and model confidence scores.
* **Visual Snapshots:** Still images (.png) of your face captured during moments of high emotional intensity.

### 2. Purpose of Processing
The data is collected strictly for **Market Research and Sentiment Analysis**. It is used to understand consumer reactions to specific stimuli and to improve the accuracy of the underlying machine learning model.

### 3. Storage and Transmission
* **Local Storage:** Data is initially stored within the application's local directory (`screenshots/` and `research_analytics_log.csv`).
* **Cloud Synchronization:** Encrypted data may be synchronized to a secure, private **AWS S3** bucket or **GitHub LFS** repository managed by the Research Department.
* **No Live Streaming:** This application **does not** stream live video to the internet. All AI inference happens locally on your hardware.

### 4. Data Distribution & Third Parties
* Your data will **never** be sold to third-party advertisers.
* Access to captured images is restricted to authorized Research Analysts for the purpose of "Human-in-the-Loop" validation.
* Aggregated, anonymized results (e.g., "80% of participants felt 'Surprise' during the ad") may be shared with project stakeholders.

### 5. Participant Rights
By proceeding with the use of this software, you provide **explicit consent** for the Research Department to store and analyze these snapshots. If you wish to withdraw your data, please contact the study administrator before the research files are finalized.

---

## 🛠 Information for Marketing Analysts
The system is designed for **Human-in-the-Loop (HITL)** validation.

### 🔍 How to Audit the Data
1.  Open `research_analytics_log.csv` in Excel.
2.  Review the corresponding image in the `screenshots/` folder.
3.  In the `true_label` column, replace `pending_review` with the actual emotion you observe. This "Ground Truth" is used to generate the final **Project Accuracy Report**.

---

## ⚠️ Troubleshooting
* **Camera Error:** Ensure no other application (Zoom, Teams, etc.) is currently using the webcam.
* **Performance:** For the best experience, ensure the laptop is connected to a power source.