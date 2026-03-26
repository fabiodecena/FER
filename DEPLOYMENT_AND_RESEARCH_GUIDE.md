# 🛠 Project Sentiment: Internal Deployment & Research Guide (v1.0)

## 📌 Executive Summary
This project delivers a standalone **Facial Expression Recognition (FER)** system using a **ConvNeXt-Tiny** backbone. Specifically engineered for the Marketing Department, it features a "Ghost UI" to eliminate participant bias and a structured **Human-in-the-Loop (HITL)** validation workflow for high-fidelity consumer analytics.

---

## 🏗 System Architecture
* **Core Model:** ConvNeXt-Tiny (Pre-trained on ImageNet-1K, Fine-tuned on AffectNet/KDEF).
* **Inference Engine:** PyTorch 2.x / OpenCV 4.13.0+.
* **Deployment Profile:** Standalone Win64 Binary (compiled via PyInstaller).
* **Data Pipeline:** Local CSV Logging → Optional AWS S3 Sync → Manual Expert Audit.

---

## 🚀 Deployment Instructions (How to Build the EXE)

If you modify the source code or update the model weights, follow these steps to recompile the portable application:

### 1. Environment Setup
Ensure your virtual environment is active and core dependencies are installed:
```powershell
pip install torch torchvision opencv-python boto3 pyinstaller pandas scikit-learn
```

### 2. Assets Preparation
Ensure the following files are in your root directory:
- `inference_webcam.py` (Main entry point)
- `model.py` (Architecture definition)
- `haarcascade_frontalface_default.xml` (Local copy of the face detector)
- `Merged/checkpoints/merged_best_20260305_162825.pt` (Trained weights)

### 3. Compilation Command
Run the following in PowerShell (formatted for Windows paths):

```powershell
pyinstaller --noconfirm --onedir --windowed `
--add-data "Merged/checkpoints/merged_best_20260305_162825.pt;Merged/checkpoints" `
--add-data "model.py;." `
--add-data "haarcascade_frontalface_default.xml;." `
inference_webcam.py
```
The portable application will be generated in `dist/inference_webcam/`.

---

## 📈 Research Workflow & Data Audit

### 1. Real-Time Data Collection
When the .exe is launched, it monitors the subject. If the AI detects an emotion with >85% confidence, it triggers an automated capture.

- **Ghost UI:** The participant sees only a clean mirror feed to ensure natural reactions.
- **Cooldown:** The system enforces a 5-second delay between captures to prevent redundant data ingestion.

### 2. Manual Validation (The "HITL" Step)
To ensure research integrity, an analyst must verify the AI's findings:

- Navigate to the `dist/inference_webcam/` directory.
- Open `research_analytics_log.csv` (Excel/Google Sheets).
- Cross-reference the `image_path` with the corresponding `.png` in the `screenshots/` folder.
- Replace the `pending_review` placeholder in the `true_label` column with the observed ground-truth emotion.

### 3. Automated Performance Analysis
Once audited, run the analysis tool to generate the final report:

```powershell
python analyze_research_data.py --csv dist/inference_webcam/research_analytics_log.csv --plot
```

---

## 🛠 S3 Cloud Ingestion Guide

## ☁️ Setting up the Cloud
1. **Create an S3 Bucket:** Name it `marketing-sentiment-data`.
2. **IAM User:** Create a "Programmatic Access" user in AWS with `AmazonS3FullAccess`.
3. **Environment Variables:** Before building the EXE, set these on your machine:
   * `AWS_KEY`: (Your Access Key)
   * `AWS_SECRET`: (Your Secret Key)

## 📥 Data Retrieval
The system automatically pushes data to the cloud in real-time. To collect all participant data at the end of the study:
1. Install the [AWS CLI](https://aws.amazon.com/cli/).
2. Run: `aws s3 sync s3://marketing-sentiment-data ./Final_Audit_Folder`
3. All images and the master CSV will be downloaded for your review.

## 🔒 Security & Privacy Compliance

- **Local Inference:** No raw video stream leaves the local hardware. Only anonymized snapshots and metadata are recorded.
- **IP Protection:** Model weights and proprietary logic are encapsulated within the binary to protect company intellectual property.
- **Informed Consent:** All participants must be provided with the `README_PARTICIPANT.md` document, which outlines the Data Consent Agreement and Privacy Policy included in the distribution package.

---

## 👨‍💻 Maintenance & Support

- **Adjusting Sensitivity:** Modify `CONF_THRESHOLD` in `inference_webcam.py` to change the trigger for automated captures.
- **Updating Labels:** If retraining the model for new emotions, ensure the `classes` list in the checkpoint matches the deployment script.
