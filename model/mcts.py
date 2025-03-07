from torch import autocast
from lib import *
from config import *
import math
import chess


class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0
        self.action = None
        

        

class MCTS:
    def __init__(self, model, device, simulations=500,batch_size=64):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct_base = 1.25
        self.c_puct = 1.2
        self.transposition_table = {}
        self.max_depth =60  # Deeper search
        self.dirichlet_alpha = 0.03

        self.batch_size = batch_size  # New batch parameter
        self.pending_nodes = []       # Nodes needing evaluation
        self.pending_paths = []       # Corresponding search paths

        self.model.eval()


    def _get_zobrist_hash(self, board):
        """Universal Zobrist hash implementation"""
        try:
            # For python-chess >= 0.28
            return board.zobrist_hash()
        except AttributeError:
            # Fallback for older versions
            return hash(board.fen())
        

    def search(self, root_board):
        root = MCTSNode(root_board)
        self._adjust_search_params(root_board)
        self.transposition_table = {}
        
        simulations_remaining = self.simulations

        # Initial expansion with transposition check
        current_hash = self._get_zobrist_hash(root.board)
        # In search loop expansion:

        if current_hash in self.transposition_table:
            policy, (stored_value, stored_turn) = self.transposition_table[current_hash]
            current_turn = root.board.turn
            value = stored_value if stored_turn == current_turn else -stored_value
            self._expand_node_with_policy(root, policy)
        else:
            self.pending_nodes.append(root)
            self.pending_paths.append([root])
            self._process_pending_batch()
            simulations_remaining -= len(self.pending_nodes)

        while simulations_remaining > 0:
            node = root
            search_path = []
            
            # Selection phase
            while node is not None and node.children and len(search_path) < self.max_depth:
                search_path.append(node)
                node = self._select_child(node)

            # Check for transposition table hit
            current_hash = self._get_zobrist_hash(node.board)
            if current_hash in self.transposition_table:
                policy, (stored_value, stored_turn) = self.transposition_table[current_hash]
                current_turn = node.board.turn
                value = stored_value if stored_turn == current_turn else -stored_value
                self._backpropagate(search_path, value)
                            
            # Terminal node handling
            if node.board.is_game_over():
                search_path.append(node)
                value = self._terminal_value(node.board)
                self._backpropagate(search_path, value)
                simulations_remaining -= 1
                continue

            # In search loop:
            while not node.children and len(search_path) < self.max_depth:
                current_hash = self._get_zobrist_hash(node.board)
                if current_hash in self.transposition_table:
                    policy, (stored_value, stored_turn) = self.transposition_table[current_hash]
                    current_turn = node.board.turn
                    value = stored_value if stored_turn == current_turn else -stored_value
                    self._expand_node_with_policy(node, policy)
                    self._backpropagate(search_path + [node], value)
                    break
                else:
                    # Add to pending batch
                    self.pending_nodes.append(node)
                    self.pending_paths.append(search_path.copy())
                    break
                
            # Process batch when ready
            if len(self.pending_nodes) >= self.batch_size or simulations_remaining <= 0:
                batch_size = min(len(self.pending_nodes), simulations_remaining)
                self.pending_nodes = self.pending_nodes[:batch_size]
                self.pending_paths = self.pending_paths[:batch_size]
                
                self._process_pending_batch()
                simulations_remaining -= batch_size

        return root

    def _process_pending_batch(self):
        """Batch evaluate all pending nodes and update tree"""
        # Prepare batch inputs
        states = [encode_board(n.board) for n in self.pending_nodes]
        states_t = torch.stack(states).to(self.device)
        
        # Batch inference
        with torch.no_grad(), autocast(device_type=self.device.type):
            policies, values = self.model(states_t)
        
        # Update nodes and transposition table
        for i, node in enumerate(self.pending_nodes):
            current_hash = self._get_zobrist_hash(node.board)
            policy = policies[i].cpu().numpy()
            value = values[i].item()
            
            # Store in transposition table
            turn = node.board.turn  # chess.WHITE/chess.BLACK
            self.transposition_table[current_hash] = (policy, (value, turn))
            
            # Expand node with batch results
            self._expand_node_with_policy(node, policy)
            
            # Backpropagate through stored path WITH VALUE
            updated_path = self.pending_paths[i] + [node]
            self._backpropagate(updated_path, value)  # Now includes current node
        
        # Reset pending buffers
        self.pending_nodes = []
        self.pending_paths = []

    def _expand_node_with_policy(self, node, policy):
        """Modified expansion using pre-computed policy"""
        legal_moves = [m.uci() for m in node.board.legal_moves if m.uci() in MOVE_VOCAB]
        
        # Create policy distribution
        legal_mask = np.zeros(len(MOVE_VOCAB), dtype=np.float32)
        for move in legal_moves:
            legal_mask[MOVE_VOCAB[move]] = 1.0
            
        policy = policy * legal_mask
        policy_sum = policy.sum()
        if policy_sum <= 0:
            policy = legal_mask.copy()
            policy_sum = policy.sum()
        policy /= policy_sum

        # Create children nodes
        for move in legal_moves:
            child = MCTSNode(node.board.copy(), parent=node)
            child.action = chess.Move.from_uci(move)
            child.prior = policy[MOVE_VOCAB[move]]
            child.board.push(child.action)
            node.children.append(child)

        # Add Dirichlet noise to root node
        if node.parent is None and node.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))
            for i, child in enumerate(node.children):
                child.prior = 0.75 * child.prior + 0.25 * noise[i]

    # In MCTS._select_child:
    def _select_child(self, node):

        for child in node.children:
            if child.board.is_checkmate():
                return child
        total_visits = sum(c.visits for c in node.children)
        if total_visits == 0:
            return random.choice(node.children)
        
        best_score = -np.inf
        best_child = None
        
        for child in node.children:
            if child.visits == 0:
                ucb = child.prior * math.sqrt(total_visits + 1)
            else:
                exploitation = child.value / child.visits
                exploration = self.c_puct * child.prior * math.sqrt(total_visits) / (child.visits + 1)
                ucb = exploitation + exploration
                
            if ucb > best_score:
                best_score = ucb
                best_child = child
                
        return best_child
    
    def _adjust_search_params(self, board):
       
        game_phase = min(1.0, len(board.move_stack) / 30)  # 0-1 (opening to endgame)
        self.c_puct = self.c_puct_base * (0.8 + 0.4 * (1 - game_phase))  # 1.5-2.25

    def _terminal_value(self, board):

        if board.is_checkmate():
            # Return +1 if current player delivered mate
            return 1.0 if board.turn != board.result() == "1-0" else -1.0
        return 0.0

    def _backpropagate(self, search_path, value):
        # Flip value only for non-terminal nodes
        for node in reversed(search_path):
            node.visits += 1
            node.value += (value - node.value) / node.visits
            if not node.board.is_game_over():
                value = -value  # Only flip for non-terminal states