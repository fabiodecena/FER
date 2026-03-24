# 😃 Facial Expression Recognition (FER) Application

## Overview

This project delivers a robust, privacy-focused **Facial Expression Recognition (FER)** system designed for both **research** and **marketing analytics**.  
Features include a high-accuracy deep learning model, "Ghost UI" for bias-free interaction, human-in-the-loop validation, and strong privacy and ethical safeguards.

---

## 📦 Project Components

- **Executable Application:**  
  A standalone `.exe` file for Windows, generated with PyInstaller, requiring no Python installation by end users.
- **Model Weights:**  
  State-of-the-art ConvNeXt-Tiny checkpoint, fine-tuned for emotion recognition.
- **Python Source Code:**  
  All scripts required for model inference, evaluation, and data handling.
- **Documentation:**
  - `DEPLOYMENT_AND_RESEARCH_GUIDE.md` &mdash; for internal, technical, and marketing/research staff.
  - `README_PARTICIPANT.md` &mdash; for end users/participants who take part in studies.

---

## 🗂 File Structure

```
.
├── dist/
│   └── inference_webcam.exe           # Main application for standalone deployment
│   └── README_PARTICIPANT.md          # Participant-facing consent/document (bundled with .exe)
├── Merged/
│   └── checkpoints/                   # Model weights
│       └── merged_best_*.pt
├── inference_webcam.py                # Main app entry point (Python)
├── model.py                           # Model architecture
├── analyze_research_data.py           # Research data analytics script
├── DEPLOYMENT_AND_RESEARCH_GUIDE.md   # Internal usage, research, and marketing doc
├── README_PARTICIPANT.md              # End-user/participant consent and instructions
└── README.md                          # THIS FILE (project-wide info)
```

---

## 🚀 Quick Start Guide

### How to Use the App (For End Users & Researchers)
1. **Launch the Application**
   - Double-click `inference_webcam.exe` in the `dist/` folder.

2. **Participate in a Session**
   - The camera feed will appear in “mirror mode.”
   - No direct prompts are given; the system records snapshots only when it detects a valid emotion above the confidence threshold.

3. **Reviewing the Data**
   - The app logs all captures to a `.csv` for later auditing by research staff.

4. **Consent**
   - All participants must read and agree to the terms in `README_PARTICIPANT.md` before starting. This file is present alongside the application for easy access and review.

---

## 📚 Documentation Overview

### `DEPLOYMENT_AND_RESEARCH_GUIDE.md` (Internal/Marketing Use)

- **Audience:** Marketing and research team; system administrators
- **Contents:**
  - System architecture and dependencies
  - Deployment instructions (how to recompile, update model, or audit data)
  - Research workflow and cloud sync options
  - Privacy and support guidelines for system maintainers

### `README_PARTICIPANT.md` (User/Participant Use)

- **Audience:** Study participants/end users
- **Contents:**
  - Simple, friendly instructions for participation
  - Details of what is recorded and when
  - Transparent privacy and consent agreement in checklist form
  - Contact details for support or withdrawal

#### **Separation Logic**
- **DEPLOYMENT_AND_RESEARCH_GUIDE.md**  
  *is for internal, technical, or marketing staff only* and should **not** be given to study subjects or consumers.
- **README_PARTICIPANT.md**  
  *is intended for participants* and must be included and shown before any data collection to ensure informed consent and transparency.

Both files are distributed with the application, but only the appropriate audience should be directed to each.

---

## 🔐 Security & Privacy Highlights

- **No raw video leaves the device**
- **Images are anonymized**; never stored with personal info
- **Participants may withdraw or request deletion at any time**
- **Only aggregate analytics are published; no individual data leaves research context**

---

## 🛠️ Maintenance & Customization

- To fine-tune thresholds, add new emotions, or change model weights, see `DEPLOYMENT_AND_RESEARCH_GUIDE.md`.
- For feedback, bug reports, or support, contact the research/engineering lead.

---

## 📄 Licensing and Attribution

- This project is for internal research and ethical marketing analytics only.
- For academic/commercial licensing, please contact the team.

---

## 🤝 Contact

- **Research Lead / Support:** [contact@email.com]

---

**Thank you to all participants and staff—for ensuring ethical, impactful research!**