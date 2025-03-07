from lib import *
from config import *
from gameplay.play import choose_move
from draw_board import draw_board
from model.chess_supervised import ChessSupervised

def human_move(board):
    """Allow human player to input a move in algebraic notation"""
    move = input("Enter your move (e.g., e2e4, Nf3): ").strip()
    try:
        chess_move = board.push_san(move)
        return chess_move
    except ValueError:  
        print("Invalid move, please try again.")
        return None

def play_human_vs_ai(model_path="chess_rl.pth", use_stockfish=False,use_mcts=True,simulations=500):
    """Allow human to play against the supervised learning AI with optional Stockfish comparison"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    color_choice = input("Choose your color (w/b): ").lower()
    while color_choice not in ['w', 'b']:
        print("Invalid choice. Please enter 'w' for white or 'b' for black.")
        color_choice = input("Choose your color (w/b): ").lower()
    
    human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
    ai_color = not human_color
    
    # Load model
    model = ChessSupervised().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    board = chess.Board()
    
    while not board.is_game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        screen.fill(COLORS['WHITE'])
        draw_board(screen, board)
        pygame.display.flip()
        
        if board.turn == human_color:
            move = human_move(board)
            if move is None:
                continue
        else:
            move = choose_move(
                board,
                model,
                device,
                use_stockfish=use_stockfish,
                use_mcts=use_mcts,simulations=simulations)
            
            print(f"AI plays: {board.san(move)}")
            board.push(move)
        
        clock.tick(FPS)
    
    pygame.quit()
    print(f"Game Over: {board.result()}")
