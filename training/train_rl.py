from lib import *
from config import *
from model.chess_supervised import ChessSupervised
from model.chess_rl import rl_train_loop

def train_reinforcement():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ChessSupervised().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=0.0003,  # Lower learning rate for fine-tuning
                                weight_decay=0.0001)
    # Try loading latest RL checkpoint first
    try:
        model.load_state_dict(torch.load("chess_rl.pth"))
        optimizer.load_state_dict(torch.load("rl_optimizer.pth"))
        print("🔥 Resuming RL training from existing checkpoint")
    except FileNotFoundError:
        try:  # Fallback to SL model
            model.load_state_dict(torch.load("chess_supervised.pth"))
            print("🔥 Starting fresh RL training with SL weights")
        except FileNotFoundError:
            print("❌ Error: Need either chess_rl.pth or chess_supervised.pth")
            return
    
    # Number of parameters
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Start RL training
    print("Starting RL/MCTS training...")
    rl_train_loop(
        model=model,
        optimizer=optimizer,
        device=device,
        episodes=5000  # Adjust as needed
    )

