import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from .model import AITextDetector
from .dataset import AIData

def run_training(train_df, eval_df, processor, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AITextDetector(vocab_size=processor.vocab_size).to(device)
    
    t_loader = DataLoader(AIData(train_df['text'], train_df['generated'], processor), batch_size=64, shuffle=True)
    e_loader = DataLoader(AIData(eval_df['text'], eval_df['generated'], processor), batch_size=64)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        for x, y in t_loader:
            x, y = x.to(device), y.to(device).view(-1, 1)
            optimizer.zero_grad(); criterion(model(x), y).backward(); optimizer.step()
        print(f"Epoch {epoch+1} complete.")

    torch.save(model.state_dict(), '/content/AIDetector/weights/text_detector.pth')
    
    # --- COMPREHENSIVE EVALUATION ---
    print("\n" + "="*30 + "\n📊 FINAL TEST SPLIT ANALYSIS\n" + "="*30)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in e_loader:
            outputs = model(x.to(device))
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            
    # Calculate Metrics
    report = classification_report(all_labels, all_preds, target_names=['Human', 'AI'])
    matrix = confusion_matrix(all_labels, all_preds)
    
    print(report)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(matrix, index=['Actual Human', 'Actual AI'], columns=['Pred Human', 'Pred AI']))
