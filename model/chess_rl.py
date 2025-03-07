from functools import partial
import torch
import numpy as np
from config import *
from lib import *
from collections import deque
from torch import GradScaler
from torch import autocast
import time
from model.replay_buffer import *
from model.mcts import MCTS
from utils.data_tools import *
import concurrent.futures
from model.chess_supervised import ChessSupervised
from concurrent.futures import as_completed

def safe_normalize(probs, eps=1e-8):
    """Improved normalization with numerical stability"""
    probs = np.nan_to_num(probs, nan=0.0)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    
    if total <= eps:
        return np.ones_like(probs)/len(probs) if len(probs) > 0 else probs
        
    probs /= total
    # Ensure sum=1.0 exactly using largest element
    idx = np.argmax(probs)
    probs[idx] += 1.0 - probs.sum()
    return probs

def run_parallel_games(model, device, num_games, max_moves=150):
    """Run multiple games in parallel using ProcessPoolExecutor"""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create partial function with serialized model
        model_cpu = copy.deepcopy(model).cpu().state_dict()
        play_fn = partial(play_single_game_wrapper, 
                        model_state=model_cpu,
                        device_str=str(device),
                        max_moves=max_moves)
        
        futures = [executor.submit(play_fn) for _ in range(num_games)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return results

def play_single_game_wrapper(model_state, device_str, max_moves):
    """Initialize model in each process"""
    device = torch.device(device_str)
    model = ChessSupervised().to(device)
    model.load_state_dict(model_state)
    return play_single_game(model, device, max_moves)

def play_single_game(model, device, max_moves=150):
    """Play a single game and return training data"""
    game_start=time.time()
    board = chess.Board()
    game_history = []
    move_count = 0
    move_list = []

    while not board.is_game_over() and move_count < max_moves:
        mcts = MCTS(model, device)
        root = mcts.search(board)
        if not root.children:
            break

        temperature = 1.0 / (1.0 + move_count/15)
        visits = np.array([max(c.visits, 1) for c in root.children], dtype=np.float32)
        logits = visits / temperature
        policy_target = safe_normalize(logits)
        
        selected_idx = np.random.choice(len(root.children), p=policy_target)
        move_uci = root.children[selected_idx].action.uci()
        move_list.append(move_uci)
        state = encode_board(board)

        board.push(root.children[selected_idx].action)
        move_count += 1

        full_policy = np.zeros(len(MOVE_VOCAB), dtype=np.float32)
        for child, prob in zip(root.children, policy_target):
            if child.action:
                full_policy[MOVE_VOCAB[child.action.uci()]] = prob
        game_history.append((state.clone(), full_policy.copy()))
        
    def calculate_z(board):
            """Simplified value targets based on game outcome"""
            if board.is_game_over():
                result = board.result()
                if result == '1-0':
                    return 1.0 
                elif result == '0-1':
                    return - 1.0 
                return 0.0
            else:
                final_mcts = MCTS(model, device, simulations=100)
                final_root = final_mcts.search(board)
                return final_root.value / max(final_root.visits, 1)
            
    z = calculate_z(board)
    updated_history = process_game_history(game_history, z)

    duration = time.time() - game_start
    result = board.result() if board.is_game_over() else "*"
    
    return (
        updated_history,
        {
            'duration': duration,
            'result': result,
            'move_count': move_count,
            'move_list': move_list.copy()
        }
    )



def process_game_history(game_history, z):
    """Process and augment game history with perspective flipping"""
    updated_history = []
    for state, policy in game_history:
        is_white = state[12,0,0] == 1
        updated_z = z if is_white else -z
        
        updated_history.append((state, policy, updated_z))

        if random.random() < 0.5:
            piece_channels = state[:12, :, :]
            turn_channel = state[12:, :, :]
            flipped_pieces = torch.flip(piece_channels, [1])
            flipped_state = torch.cat([flipped_pieces, turn_channel], dim=0)
            
            flipped_policy = np.zeros_like(policy)
            for src, dst in VERTICAL_MIRROR.items():
                flipped_policy[dst] = policy[src]
            updated_history.append((flipped_state, flipped_policy, updated_z))
    
    return updated_history

def rl_train_loop(model, optimizer, device, episodes=1000, batch_size=4096,
                 parallel_games=4, games_per_update=64):
    replay_buffer = load_rl_buffer()
    if len(replay_buffer) > 1_000_000:
        replay_buffer = deque(list(replay_buffer)[-1_000_000:])

    print(f"Initial Replay Buffer Size: {len(replay_buffer)}")
    scaler = GradScaler()
    accum_steps = 4
    total_time_for_10 = 0
    games_since_last_save = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=10_000,
        eta_min=0.00001
    )
    
    # Setup logging
    log_dir = "training_games"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"games_{timestamp}.txt")

    games_played = 0
    batch_counter = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_games) as executor:
        # Initialize with first batch of games
        model_cpu = copy.deepcopy(model).cpu().state_dict()
        play_fn = partial(play_single_game_wrapper, 
                        model_state=model_cpu,
                        device_str=str(device),
                        max_moves=150)
        
        # Create initial futures
        futures = [executor.submit(play_fn) for _ in range(min(parallel_games, episodes))]
        
        while games_played < episodes:
            
            # Process completed games
            for future in concurrent.futures.as_completed(futures):
                try:
                    game_data, metadata = future.result()
                    replay_buffer.extend(game_data)
                    games_played += 1
                    games_since_last_save += 1

                    # Log individual game results
                    duration = metadata['duration']
                    result = metadata['result']
                    move_count = metadata['move_count']
                    total_time_for_10 += duration

                    mins, secs = divmod(duration, 60)
                    print(f"Game {games_played}: {result} in {move_count//2} moves"
                          f"({int(mins)}m {int(secs)}s)|"
                          f"Buffer:{len(replay_buffer)}|",end='')

                    # Save game to file
                    with open(log_file, "a") as f:
                        f.write(f"Game {games_played}: {result} | Moves: {' '.join(metadata['move_list'])}\n")
                    # Training update
                    if len(replay_buffer) >= batch_size:
                        optimizer.zero_grad()
                        total_loss = 0.0
                        policy_loss_total = 0.0
                        value_loss_total = 0.0
                        grad_norms = []
                        
                        for accum_step in range(accum_steps):
                            mini_batch = random.sample(replay_buffer, batch_size//accum_steps)
                            states, policies, values = zip(*mini_batch)
                            
                            states_t = torch.stack(states).to(device)
                            policies_t = torch.tensor(np.stack(policies), device=device, dtype=torch.float32)
                            values_t = torch.clamp(torch.tensor(values, device=device, dtype=torch.float32), -1.0, 1.0)
                            
                            with autocast(device_type=device.type):
                                policy_pred, value_pred = model(states_t)
                                policy_logprobs = F.log_softmax(policy_pred, dim=1)
                                policy_loss = F.kl_div(policy_logprobs, policies_t, reduction='batchmean')
                                value_loss = F.huber_loss(value_pred.squeeze(), values_t)
                                entropy = -(policy_pred.softmax(dim=1) * policy_logprobs).sum(1).mean()
                                loss = (policy_loss + 0.5* value_loss + 0.1*entropy) / accum_steps
                            
                            scaler.scale(loss).backward()
                            total_loss += loss.item()
                            policy_loss_total += policy_loss.item() / accum_steps
                            value_loss_total += value_loss.item() / accum_steps

                        # Gradient clipping and logging
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                        
                        scaler.step(optimizer)
                        scheduler.step()
                        scaler.update()

                        print(f" Grad norms:{np.mean(grad_norms):.2f} ± {np.std(grad_norms):.2f}"
                            f" | Total Loss: {total_loss:.4f} | Policy Loss: {policy_loss_total:.4f}"
                            f" | Value Loss: {value_loss_total:.4f}")
                        batch_counter += 1

                    # Checkpoint saving
                    if games_played % 10 == 0:
                        save_checkpoint(model, optimizer, replay_buffer, games_played)


                except Exception as e:
                    print(f"⚠️ Game failed: {str(e)}")
                
                # Submit new game if we haven't reached the limit
                if games_played < episodes:
                    current_model = copy.deepcopy(model).cpu().state_dict()
                    futures.append(
        executor.submit(
            play_single_game_wrapper,
            model_state=current_model,  # Updated weights
            device_str=str(device),
            max_moves=150
        )
    )

            # # Periodic summaries
            # if games_since_last_save == 10:
            #     total_mins, total_secs = divmod(total_time_for_10, 60)
            #     avg_time = total_time_for_10 / games_since_last_save
            #     print(f"\n⏱ Last {games_since_last_save} games: {int(total_mins)}m {int(total_secs)}s "
            #           f"(avg {avg_time:.1f}s/game)")
            #     total_time_for_10 = 0
            #     games_since_last_save = 0


    print("\n🎉 Training complete!")
    save_checkpoint(model, optimizer, replay_buffer, games_played)

def save_checkpoint(model, optimizer, replay_buffer, games_played):
    """Save training state"""
    torch.save(model.state_dict(), f"chess_rl.pth")
    torch.save(optimizer.state_dict(), f"rl_optimizer.pth")
    save_rl_buffer(replay_buffer)
    print(f"💾 Saved checkpoint after {games_played} games")

