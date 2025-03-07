from lib import *
from config import *
from model.chess_supervised import ChessSupervised
from draw_board import draw_board
from gameplay.play import choose_move

def play_ai_vs_ai(model_path="chess_supervised.pth", use_stockfish=False, use_mcts=True, simulations=500):
    """AI vs AI with MCTS support"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
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
        
        # Both players use MCTS
        move = choose_move(
            board, model, device,
            use_stockfish=use_stockfish,
            use_mcts=use_mcts,
            simulations=simulations
        )
        print(f"AI plays: {board.san(move)}")
        board.push(move)
        
        clock.tick(FPS)
    
    pygame.quit()
    print(f"Game Over: {board.result()}")