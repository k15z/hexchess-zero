from __future__ import annotations
"""Play arena games between two MCTS agents."""

try:
    import hexchess
except ImportError:
    hexchess = None


def play_arena_game(
    simulations: int,
    new_goes_first: bool,
    new_model_path: str | None = None,
    old_model_path: str | None = None,
) -> str:
    """
    Play one arena game between two MCTS agents.

    Args:
        simulations: Number of MCTS simulations per move.
        new_goes_first: If True, the "new" model plays White.
        new_model_path: ONNX model for the new agent (None = random).
        old_model_path: ONNX model for the old/best agent (None = random).

    Returns:
        "new", "old", or "draw"
    """
    if hexchess is None:
        raise ImportError("hexchess bindings not available")

    game = hexchess.Game()
    search_new = hexchess.MctsSearch(simulations=simulations, model_path=new_model_path)
    search_old = hexchess.MctsSearch(simulations=simulations, model_path=old_model_path)

    move_count = 0
    max_moves = 500  # safety limit

    while not game.is_game_over() and move_count < max_moves:
        is_white = game.side_to_move() == "white"
        # Decide which search engine to use
        if (is_white and new_goes_first) or (not is_white and not new_goes_first):
            search = search_new
        else:
            search = search_old

        result = search.run(game, temperature=0.1)
        best_move = result["best_move"]

        game.apply_move(
            best_move["from_q"],
            best_move["from_r"],
            best_move["to_q"],
            best_move["to_r"],
            best_move.get("promotion"),
        )
        move_count += 1

    status = game.status()

    if status == "checkmate_white":
        winner_is_white = True
    elif status == "checkmate_black":
        winner_is_white = False
    else:
        return "draw"

    # Map winner color back to new/old
    if new_goes_first:
        return "new" if winner_is_white else "old"
    else:
        return "new" if not winner_is_white else "old"
