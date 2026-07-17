import gradio as gr
import torch
from .model import AITextDetector

def launch_app(processor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model structure
    model = AITextDetector(vocab_size=processor.vocab_size).to(device)
    
    # Load the trained weights from your training run
    try:
        model.load_state_dict(torch.load('/content/AIDetector/weights/text_detector.pth', map_location=device))
        model.eval()
        print("✅ Model weights loaded. UI starting...")
    except Exception as e:
        print(f" Could not load weights: {e}. Ensure training finished first.")

    def detect(text):
        if not text.strip():
            return "Please enter some text to analyze."
            
        # Convert text to tokens and move to device
        tokens = processor.transform(text).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            prob = model(tokens).item()
            
        label = "AI GENERATED" if prob > 0.5 else "HUMAN WRITTEN"
        confidence = prob if prob > 0.5 else (1 - prob)
        
        return f"{label} ({confidence*100:.2f}% Confidence)"

    # Define the Gradio Interface
    interface = gr.Interface(
        fn=detect,
        inputs=gr.Textbox(lines=10, label="Input Text", placeholder="Paste paragraph here..."),
        outputs=gr.Textbox(label="Analysis Result"),
        title="🛡️ AI Text Guardian",
        description="This tool uses a Bidirectional LSTM to detect linguistic patterns typical of AI generation."
    )
    
    interface.launch(share=True)
