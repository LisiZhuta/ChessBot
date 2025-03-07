from lib import *
from config import *
from model.stockfish_helper import StockfishHelper
from model.chess_rl import MCTS 

stockfish = StockfishHelper()

def choose_move(board, model, device, use_stockfish=False, use_mcts=True, simulations=800):
    """Pure exploitation move selection"""
    legal_moves = list(board.legal_moves)
    
    if use_stockfish:  # Stockfish as fallback
        if sf_move := stockfish.get_best_move(board):
            return chess.Move.from_uci(sf_move)
    
    if use_mcts:
        mcts = MCTS(model, device, simulations=simulations)
        root = mcts.search(board)
        ai_move = max(root.children, key=lambda c: c.visits).action  # Pure exploitation
    else:
        with torch.no_grad():
            state = encode_board(board).unsqueeze(0).to(device)
            policy, _ = model(state)
        
        # Direct argmax without temperature/noise
        move_idx = torch.argmax(policy[0][[MOVE_VOCAB[m.uci()] for m in legal_moves]])
        ai_move = legal_moves[move_idx.item()]
    
    return ai_move