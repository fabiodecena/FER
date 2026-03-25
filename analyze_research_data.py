"""
analyze_research_data.py - Marketing Analytics Edition
FULL SUITE: Distribution, Timeline, Discrepancy, and Accuracy.
"""

import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def analyze_research_log(csv_path, plot_cm=False):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    # Load Data
    df = pd.read_csv(csv_path)

    # --- Configuration ---
    pred_col = 'ai_predicted_emotion'
    true_col = 'true_label'
    conf_col = 'ai_confidence'

    # Normalize data for logic
    df[pred_col] = df[pred_col].str.lower()
    if true_col in df.columns:
        df[true_col] = df[true_col].str.lower()

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- COLOR PALETTE (RGB Synced) ---
    emotion_colors = {
        'angry': '#FF0000',  # Red
        'disgust': '#808000',  # Olive
        'fear': '#800080',  # Purple
        'happy': '#00FF00',  # Green
        'neutral': '#C8C8C8',  # Light Gray
        'sad': '#0000FF',  # Blue
        'surprise': '#FFFF00'  # Yellow
    }

    print("--- 📊 Marketing Research Summary ---")
    print(f"Total Samples in Log: {len(df)}")

    # 1. Distribution Console Output
    print("\nAI Prediction Distribution (All Samples):")
    dist = df[pred_col].value_counts()
    for emo, count in dist.items():
        print(f"  {emo.capitalize():10}: {count} frames")

    if plot_cm:
        # Prepare plotting data
        df_plot = df.copy()
        df_plot['Emotion'] = df_plot[pred_col].str.capitalize()
        pretty_palette = {k.capitalize(): v for k, v in emotion_colors.items()}

        # --- GRAPH 1: Pie Chart (Overall Distribution) ---
        plt.figure(figsize=(8, 8))
        pie_data = df_plot['Emotion'].value_counts()
        pie_colors = [pretty_palette.get(label, '#808080') for label in pie_data.index]
        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%',
                colors=pie_colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        plt.title('Overall Ad Sentiment Distribution', fontsize=14)
        plt.savefig("emotion_distribution_pie.png")

        # --- GRAPH 2: Engagement Timeline ---
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df_plot, x='timestamp', y=conf_col, hue='Emotion',
                        palette=pretty_palette, s=150, alpha=0.8, edgecolor='black')
        plt.title('Engagement Intensity Over Time', fontsize=14)
        plt.ylabel('AI Confidence')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig("engagement_timeline.png")

        # --- AUDIT FILTERS (For Human vs AI Comparison) ---
        audited_df = df[df[true_col] != "pending_review"].dropna(subset=[true_col]).copy()

        if len(audited_df) > 0:
            # --- GRAPH 3: Discrepancy Bar Chart ---
            ai_counts = audited_df[pred_col].value_counts().rename("AI Prediction")
            human_counts = audited_df[true_col].value_counts().rename("Human Verified")
            comparison_df = pd.concat([ai_counts, human_counts], axis=1).fillna(0)
            comparison_df.index = [i.capitalize() for i in comparison_df.index]

            plt.figure(figsize=(10, 6))
            comparison_df.plot(kind='bar', ax=plt.gca(), color=['#AED6F1', '#2E86C1'])
            plt.title('AI Predictions vs. Human Audit Count', fontsize=14)
            plt.ylabel('Frames')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("ai_vs_human_discrepancy.png")

            # --- GRAPH 4: Confusion Matrix (Accuracy) ---
            plt.figure(figsize=(10, 8))
            labels = sorted(list(set(audited_df[true_col].unique()) | set(audited_df[pred_col].unique())))
            cm = confusion_matrix(audited_df[true_col], audited_df[pred_col], labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[l.capitalize() for l in labels],
                        yticklabels=[l.capitalize() for l in labels])
            plt.title('Accuracy Audit Heatmap')
            plt.xlabel('AI Predicted')
            plt.ylabel('Human Verified')
            plt.savefig("research_performance_report.png")

            print(f"\n✅ Audit visualization complete ({len(audited_df)} verified samples).")
        else:
            print("\n📝 Skipping Audit Graphs: No 'pending_review' placeholders have been replaced yet.")

        print("📈 All available graphs saved to directory.")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze Marketing Research Logs.")
    parser.add_argument('--csv', type=str, default="research_analytics_log.csv")
    parser.add_argument('--plot', action='store_true', help="Generate all visual reports.")
    args = parser.parse_args()
    analyze_research_log(args.csv, args.plot)

if __name__ == "__main__":
    main()