import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def analyze_feedback_log(csv_path, show_details=False, plot_cm=False):
    df = pd.read_csv(csv_path)

    total = len(df)
    correct = (df['user_feedback'] == 'yes').sum()
    accuracy = correct / total if total > 0 else 0.0

    print(f"\nDeployment (user-validated) accuracy: {accuracy:.2%} ({correct}/{total})")

    if show_details:
        print("\nPer-class accuracy:")
        for cls in sorted(df['predicted_class'].unique()):
            cls_df = df[df['predicted_class'] == cls]
            cls_total = len(cls_df)
            cls_correct = (cls_df['user_feedback'] == 'yes').sum()
            cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
            print(f"  {cls:10}: {cls_acc:.2%} ({cls_correct}/{cls_total})")
        print()

    # --- Confusion matrix and classification report ---
    if {'true_label', 'predicted_class'}.issubset(df.columns):
        y_true = df['true_label'].values
        y_pred = df['predicted_class'].values

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        labels = sorted(df['true_label'].unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("\nConfusion Matrix:")
        print(cm)

        if plot_cm:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, cmap='Blues')
                plt.xticks(range(len(labels)), labels, rotation=45)
                plt.yticks(range(len(labels)), labels)
                plt.colorbar()
                plt.xlabel('Predicted label')
                plt.ylabel('True label')
                plt.title('User-Validated Confusion Matrix')
                plt.show()
            except ImportError:
                print("matplotlib is not installed. Please install it to plot confusion matrix.")

    else:
        print("\nFor confusion matrix/classification report, please ensure your CSV logs true_label column.")

def main():
    parser = argparse.ArgumentParser(description="Analyze user feedback log and compute deployment accuracy.")
    parser.add_argument('--csv', type=str, default="user_feedback_log.csv", help="Path to the feedback CSV file.")
    parser.add_argument('--details', action='store_true', help="Show per-class accuracy breakdown.")
    parser.add_argument('--plot', action='store_true', help="Plot confusion matrix if true labels are present.")
    args = parser.parse_args()

    analyze_feedback_log(args.csv, args.details, args.plot)

if __name__ == "__main__":
    main()