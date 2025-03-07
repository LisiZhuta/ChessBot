import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from config import *
from model.chess_supervised import ChessSupervised
from utils.data_tools import *
from collections import deque
import pickle
import time
from model.replay_buffer import *



def open_pgn_with_retry(file_path, max_retries=3, delay=2):
    """Attempt to open a PGN file with retries in case of failure."""
    attempts = 0
    while attempts < max_retries:
        try:
            return open(file_path, "r", encoding="utf-8")
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            attempts += 1
            if attempts < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to open {file_path} after {max_retries} attempts.")
                return None

def train_pgn_supervised(pgn_files, save_interval=500, log_interval=500):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ChessSupervised().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    scaler = GradScaler()

    # Load replay buffer
    replay_buffer = load_sl_buffer()

    # Number of parameters
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Replay Buffer Size : {len(replay_buffer)}")

    total_games = 0
    total_batches = 0

    for pgn_file in pgn_files:
        with open_pgn_with_retry(pgn_file) as f:
            while (game := chess.pgn.read_game(f)):
                total_games += 1
                try:
                    boards, moves, outcomes = process_game(game)
                    if not boards:
                        continue

                    # Add new data to replay buffer

                    replay_buffer.extend(zip(boards, moves, outcomes))

                   # train_pgn_supervised.py (correction)
                    if len(replay_buffer) >= MIN_REPLAY_SIZE:  # Changed condition
                        batch = random.sample(replay_buffer, BATCH_SIZE)
                        batch_boards, batch_moves, batch_outcomes = zip(*batch)  # Unpack tuples

                        batch_boards = torch.stack(batch_boards).to(device)
                        batch_moves = torch.tensor(batch_moves).to(device)
                        batch_outcomes = torch.tensor(batch_outcomes).float().to(device)
                    else:
                        continue


                    with autocast(device_type=device.type, dtype=torch.float16):
                        policy_output, value_output = model(batch_boards)
                        policy_loss = F.cross_entropy(policy_output, batch_moves, label_smoothing=0.1)
                        value_loss = F.mse_loss(value_output.view(-1), batch_outcomes.view(-1))
                        total_loss = policy_loss + 0.5 * value_loss

                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_batches += 1

                    if total_batches % log_interval == 0:
                        print(
                            f"Batch {total_batches} (Game {total_games}) | "
                            f"Policy Loss: {policy_loss.item():.4f}, "
                            f"Value Loss: {value_loss.item():.4f}, "
                            f"Total Loss: {total_loss.item():.4f}"
                        )

                    if total_batches % save_interval == 0:
                        torch.save(model.state_dict(), "chess_supervised.pth")
                        save_sl_buffer(replay_buffer)

                except Exception as e:
                    print(f"Error in game {total_games}: {str(e)}")

    torch.save(model.state_dict(), "chess_supervised.pth")
    save_sl_buffer(replay_buffer)
    print("Training Complete")
