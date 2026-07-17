import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def save_metrics(eval_df, processor, model_path='/content/AIDetector/weights/text_detector.pth', output_dir='/content/AIDetector/results'):
    # 1. Setup
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Model
    from .model import AITextDetector
    model = AITextDetector(vocab_size=processor.vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Predict
    all_preds = []
    all_labels = eval_df['generated'].tolist()
    
    with torch.no_grad():
        for text in eval_df['text']:
            tokens = processor.transform(text).unsqueeze(0).to(device)
            prob = model(tokens).item()
            all_preds.append(1 if prob > 0.5 else 0)

    # 4. Save Classification Report to .txt
    report = classification_report(all_labels, all_preds, target_names=['Human', 'AI'])
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write("=== AI Text Detector Evaluation ===\n")
        f.write(report)
    
    # 5. Save Confusion Matrix to .png
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Human', 'Pred AI'], 
                yticklabels=['Actual Human', 'Actual AI'])
    plt.title('Confusion Matrix Heatmap')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    print(f"✅ Evaluation complete! Files saved in: {output_dir}")
