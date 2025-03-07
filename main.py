from config import *
from gameplay.play_human_vs_ai import play_human_vs_ai
from gameplay.play_ai_vs_ai import play_ai_vs_ai
from training.train_pgn import train_pgn_supervised
from training.train_rl import train_reinforcement
import multiprocessing as mp

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

if __name__ == "__main__":
   
    mp.set_start_method('spawn', force=True)
    
    print("\n1. Train model with Supervised Learning + PGN DB")
    print("2. Train model with Reinforced Learning + MCTS \n")

    print("3. Watch AI vs AI | No MCTS") #False by default. Change in gameplay ai_vs_ai in function params
    print("4. Play HUMAN vs AI | No MCTS \n") #False by default. Change in gameplay human_vs_ai in function params

    print("5. Play AI vs AI | Costumized")
    print("6. Play HUMAN vs AI | Costumized\n")
    


    choice = input("Select mode: ")

    if choice =="1":
        train_pgn_supervised(["pgn/lichess_elite_2020-06.pgn"])
        exit()

    elif choice =="2":
        train_reinforcement()
        exit()

    elif choice == "3":
        play_ai_vs_ai()

    elif choice == "4":
        play_human_vs_ai()


    elif choice == "5":
        play_ai_vs_ai(
        model_path="chess_rl.pth", # Specify Path
        use_mcts=True,  # Force MCTS even with SL model
        simulations=200)

    elif choice == "6":
        play_human_vs_ai(
        model_path="chess_rl.pth", # Specify Path
        use_mcts=True,  # Force MCTS even with SL model
        simulations=500)

    else:
        print("Invalid choice")
    