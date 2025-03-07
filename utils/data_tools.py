from lib import *
from config import *


def create_mirror_maps():
    vertical_map = {}
    horizontal_map = {}
    
    for uci, idx in MOVE_VOCAB.items():
        # Vertical flip (a1 -> a8, b2 -> b7 etc)
        v_from = uci[0] + str(9 - int(uci[1]))
        v_to = uci[2] + str(9 - int(uci[3]))
        v_uci = v_from + v_to + (uci[4:] if len(uci) > 4 else '')
        
        # Horizontal flip (a1 -> h1, b2 -> g2 etc)
        h_from = chr(ord('h') - (ord(uci[0]) - ord('a'))) + uci[1]
        h_to = chr(ord('h') - (ord(uci[2]) - ord('a'))) + uci[3]
        h_uci = h_from + h_to + (uci[4:] if len(uci) > 4 else '')
        
        vertical_map[idx] = MOVE_VOCAB.get(v_uci, idx)
        horizontal_map[idx] = MOVE_VOCAB.get(h_uci, idx)
        
    return vertical_map, horizontal_map

VERTICAL_MIRROR, HORIZONTAL_MIRROR = create_mirror_maps()




def augment_board(board):
    # Random vertical/horizontal flip
    if random.random() < 0.5:
        board = board.transform(chess.flip_vertical)
    if random.random() < 0.5:
        board = board.transform(chess.flip_horizontal)
    return board

def adjust_move_for_flip(uci, flip_type):
    """Adjust UCI move for board flip augmentation"""
    def flip_square(sq, flip_type):
        file, rank = sq[0], sq[1]
        if flip_type == 'vertical':
            return file + str(9 - int(rank))
        elif flip_type == 'horizontal':
            return chr(ord('h') - (ord(file) - ord('a'))) + rank
        return sq

    from_sq = uci[:2]
    to_sq = uci[2:4]
    promo = uci[4:] if len(uci) > 4 else ''

    new_from = flip_square(from_sq, flip_type)
    new_to = flip_square(to_sq, flip_type)
    
    # Handle promotion ranks
    if new_to[1] in ('1', '8') and promo:
        promo = promo.lower() if new_to[1] == '1' else promo.upper()
    
    return new_from + new_to + promo

def process_game(game, min_elo=2000):
    """Fixed version with proper move adjustment"""
    boards, moves, outcomes = [], [], []
    
    # Elo filtering
    white_elo = int(game.headers.get("WhiteElo", 0))
    black_elo = int(game.headers.get("BlackElo", 0))
    if white_elo < min_elo or black_elo < min_elo:
        return [], [], []
    
    result = game.headers.get("Result", "0.5-0.5")
    outcome_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
    final_outcome = outcome_map.get(result, 0.0)
    
    board = game.board()
    for move in game.mainline_moves():
        try:
            if not board.is_legal(move):
                continue
                
            original_uci = move.uci()
            if original_uci not in MOVE_VOCAB:
                continue

            # Apply augmentation and adjust move
            flip_type = None
            augmented_board = board.copy()
            if random.random() < 0.5:
                augmented_board = augmented_board.transform(chess.flip_vertical)
                flip_type = 'vertical'
            if random.random() < 0.5:
                augmented_board = augmented_board.transform(chess.flip_horizontal)
                flip_type = 'horizontal' if not flip_type else flip_type + '_horizontal'

            adjusted_uci = original_uci
            if flip_type:
                adjusted_uci = adjust_move_for_flip(original_uci, flip_type)
                if adjusted_uci not in MOVE_VOCAB:
                    continue  # Skip invalid transformed moves

            current_player = 1.0 if board.turn else -1.0
            outcome = final_outcome * current_player
            
            boards.append(encode_board(augmented_board))
            moves.append(MOVE_VOCAB[adjusted_uci])
            outcomes.append(outcome)
            
            board.push(move)

        except Exception as e:
            print(f"Error processing move {move}: {str(e)}")
            continue
    
    return boards, moves, outcomes