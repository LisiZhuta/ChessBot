from stockfish import Stockfish

class StockfishHelper:
    def __init__(self, stockfish_path="stockfish"):
        self.stockfish = Stockfish(stockfish_path)
        self.stockfish.set_depth(20)  # Set search depth
        self.stockfish.set_skill_level(20)  # Maximum strength

    def get_best_move(self, board):
        """Returns Stockfish's best move for a given position"""
        self.stockfish.set_fen_position(board.fen())
        best_move = self.stockfish.get_best_move()
        return best_move

    def evaluate_position(self, board):
        """Returns Stockfish evaluation in centipawns"""
        self.stockfish.set_fen_position(board.fen())
        return self.stockfish.get_evaluation()
