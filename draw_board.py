from config import *

#function to create the board
def draw_board(screen, board):
    """Draw chess board with pieces using Pygame with modern aesthetics and external coordinates"""
    screen.fill(COLORS['WHITE'])
    
    # Draw the board grid (8x8 squares)
    for row in range(8):
        for col in range(8):
            x = MARGIN + col * SQUARE_SIZE
            y = MARGIN + row * SQUARE_SIZE
            color = COLORS['LIGHT'] if (row + col) % 2 == 0 else COLORS['DARK']
            pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
            
            # Draw the piece if there is one on this square
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                img = pygame.image.load(PIECE_IMAGES[piece.symbol()])
                img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
                screen.blit(img, (x, y))
    
    # Draw column labels (a-h) below the board
    font = pygame.font.SysFont('Arial', 24, bold=True)
    for col in range(8):
        label = chr(97 + col)  # 'a' to 'h'
        text = font.render(label, True, COLORS['BLACK'])
        # Position at the bottom, centered under each column
        x = MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_width() // 2
        y = MARGIN + 8 * SQUARE_SIZE + 10  # 10 pixels below the board
        screen.blit(text, (x, y))

    # Draw row labels (1-8) to the left of the board
    for row in range(8):
        label = str(8 - row)  # 8 to 1
        text = font.render(label, True, COLORS['BLACK'])
        # Position to the left of the board
        x = MARGIN - 30  # 30 pixels left of the board's left edge
        y = MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_height() // 2
        screen.blit(text, (x, y))

    # Draw row labels (1-8) to the right of the board
    for row in range(8):
        label = str(8 - row)
        text = font.render(label, True, COLORS['BLACK'])
        # Position to the right of the board
        x = MARGIN + 8 * SQUARE_SIZE + 10  # 10 pixels right of the board
        y = MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_height() // 2
        screen.blit(text, (x, y))

    # Draw border around the board
    pygame.draw.rect(screen, COLORS['BLACK'], 
                    (MARGIN, MARGIN, BOARD_WIDTH, BOARD_HEIGHT), 3)