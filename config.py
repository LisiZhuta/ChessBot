from lib import *

#this folder holds mostly constant variables, except for Neural Network Parameters

# Pygame Configuration
WIDTH, HEIGHT = 700, 700
MARGIN = 50
BOARD_WIDTH = WIDTH - 2 * MARGIN
BOARD_HEIGHT = HEIGHT - 2 * MARGIN
SQUARE_SIZE = BOARD_HEIGHT // 8  # 75 pixels per square
FPS = 10

COLORS = {
    'WHITE': (255, 255, 255),   
    'BLACK': (0, 0, 0),
    'LIGHT': (238, 238, 210),
    'DARK': (118, 150, 86)
}

PIECE_IMAGES = {
    'K': 'images/king_w.png', 'Q': 'images/queen_w.png', 
    'R': 'images/rook_w.png', 'B': 'images/bishop_w.png',
    'N': 'images/knight_w.png', 'P': 'images/pawn_w.png',
    'k': 'images/king_b.png', 'q': 'images/queen_b.png',
    'r': 'images/rook_b.png', 'b': 'images/bishop_b.png',
    'n': 'images/knight_b.png', 'p': 'images/pawn_b.png'
}

def encode_board(board):
    """Vectorized board encoding with turn information"""
    pieces = torch.zeros(13, 8, 8)  # Add 13th channel for turn
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            channel = piece.piece_type - 1 + (6 * (not piece.color))
            pieces[channel, sq//8, sq%8] = 1
    # Add turn information (channel 12)
    pieces[12, :, :] = 1.0 if board.turn else 0
    return pieces

def _create_move_vocab():
    """Generates vocabulary for pawn moves, non-promotions, and valid underpromotions."""
    vocab = {}
    idx = 0

    # Non-promotion moves (all piece types)
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            move = chess.Move(from_sq, to_sq)
            if move.uci() in vocab:
                continue
            vocab[move.uci()] = idx
            idx += 1

    # Valid underpromotions (pawn to 1st/8th rank)
    for from_sq in chess.SQUARES:
        from_rank = chess.square_rank(from_sq)
        for to_sq in chess.SQUARES:
            to_rank = chess.square_rank(to_sq)
            # Pawns can only promote on 7th (Black) or 8th (White) rank
            if to_rank not in (0, 7):  # 0=rank 1 (Black), 7=rank 8 (White)
                continue

            # Check if it's a pawn move (from 2nd/7th rank for promotions)
            if (abs(from_rank - to_rank) != 1) and (abs(from_rank - to_rank) != 2):
                continue  # Not a pawn move

            # Add underpromotions (queen, rook, bishop, knight)
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                if promo_move.uci() not in vocab:
                    vocab[promo_move.uci()] = idx
                    idx += 1

    return vocab

MOVE_VOCAB = _create_move_vocab()
MOVE_VOCAB_SIZE = len(MOVE_VOCAB)
LEGAL_MOVES = {m.uci() for m in chess.Board().legal_moves if m.uci() in MOVE_VOCAB}