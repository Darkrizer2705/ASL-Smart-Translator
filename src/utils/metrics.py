from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def save_model_metrics(y_true, y_pred, classes, model_name: str, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes)
    
    metrics_file = results_dir / f"{model_name}_metrics.txt"
    with metrics_file.open("w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    print(f"Saved metrics to: {metrics_file}")
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    # Adjust figure size based on number of classes
    fig_size = max(10, len(classes) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
    plt.tight_layout()
    
    cm_file = results_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_file)
    plt.close()
    
    print(f"Saved confusion matrix to: {cm_file}")
