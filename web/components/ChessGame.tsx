"use client";

import { useRef, useState, type CSSProperties } from "react";
import { Chess, type Square } from "chess.js";
import { Chessboard } from "react-chessboard";

type MoveSquares = { from: Square; to: Square };

const LAST_MOVE_STYLE: CSSProperties = { background: "rgba(255, 214, 10, 0.45)" };
const SELECTED_STYLE: CSSProperties = { background: "rgba(20, 85, 30, 0.5)" };
const CHECK_STYLE: CSSProperties = {
  background: "radial-gradient(circle, rgba(255, 0, 0, 0.8) 0%, rgba(255, 0, 0, 0) 75%)",
};
const MOVE_DOT_STYLE: CSSProperties = {
  background: "radial-gradient(circle, rgba(0, 0, 0, 0.2) 22%, transparent 24%)",
};
const CAPTURE_RING_STYLE: CSSProperties = {
  background: "radial-gradient(circle, transparent 58%, rgba(0, 0, 0, 0.2) 60%)",
};

export default function ChessGame() {
  // chess.js keeps the move history needed for threefold repetition,
  // so we hold one instance in a ref and re-render off the FEN.
  const gameRef = useRef(new Chess());
  const game = gameRef.current;
  const [fen, setFen] = useState(game.fen());
  const [selected, setSelected] = useState<Square | null>(null);
  const [lastMove, setLastMove] = useState<MoveSquares | null>(null);
  const [pendingPromotion, setPendingPromotion] = useState<MoveSquares | null>(null);
  const [orientation, setOrientation] = useState<"white" | "black">("white");

  const legalMovesFrom = (square: Square) => game.moves({ square, verbose: true });

  const isPromotion = (from: Square, to: Square) =>
    legalMovesFrom(from).some((m) => m.to === to && m.promotion);

  function makeMove(from: Square, to: Square, promotion?: string): boolean {
    try {
      game.move({ from, to, promotion });
    } catch {
      return false; // chess.js throws on illegal moves
    }
    setFen(game.fen());
    setLastMove({ from, to });
    setSelected(null);
    return true;
  }

  function onPieceDrop(from: Square, to: Square): boolean {
    return makeMove(from, to);
  }

  function onPromotionPieceSelect(piece?: string, from?: Square, to?: Square): boolean {
    const move = from && to ? { from, to } : pendingPromotion;
    setPendingPromotion(null);
    if (!piece || !move) {
      setSelected(null);
      return false;
    }
    return makeMove(move.from, move.to, piece[1].toLowerCase());
  }

  function onSquareClick(square: Square) {
    if (game.isGameOver()) return;

    if (selected && legalMovesFrom(selected).some((m) => m.to === square)) {
      if (isPromotion(selected, square)) {
        setPendingPromotion({ from: selected, to: square });
      } else {
        makeMove(selected, square);
      }
      return;
    }

    const piece = game.get(square);
    if (piece && piece.color === game.turn() && square !== selected) {
      setSelected(square);
    } else {
      setSelected(null);
    }
  }

  function newGame() {
    game.reset();
    setFen(game.fen());
    setSelected(null);
    setLastMove(null);
    setPendingPromotion(null);
  }

  const squareStyles: Record<string, CSSProperties> = {};
  if (lastMove) {
    squareStyles[lastMove.from] = LAST_MOVE_STYLE;
    squareStyles[lastMove.to] = LAST_MOVE_STYLE;
  }
  if (game.isCheck()) {
    const [kingSquare] = game.findPiece({ type: "k", color: game.turn() });
    squareStyles[kingSquare] = CHECK_STYLE;
  }
  if (selected) {
    squareStyles[selected] = SELECTED_STYLE;
    for (const m of legalMovesFrom(selected)) {
      squareStyles[m.to] = m.captured ? CAPTURE_RING_STYLE : MOVE_DOT_STYLE;
    }
  }

  return (
    <div className="flex w-full max-w-[560px] flex-col gap-4">
      <p className="text-center text-lg font-medium" aria-live="polite">
        {gameStatus(game)}
      </p>

      <Chessboard
        position={fen}
        boardOrientation={orientation}
        onPieceDrop={onPieceDrop}
        onSquareClick={onSquareClick}
        onPieceDragBegin={() => setSelected(null)}
        onPromotionCheck={(from, to) => isPromotion(from, to)}
        onPromotionPieceSelect={onPromotionPieceSelect}
        showPromotionDialog={pendingPromotion !== null}
        promotionToSquare={pendingPromotion?.to ?? null}
        isDraggablePiece={({ piece }) => !game.isGameOver() && piece[0] === game.turn()}
        customSquareStyles={squareStyles}
        customBoardStyle={{ borderRadius: "6px", boxShadow: "0 4px 16px rgba(0, 0, 0, 0.3)" }}
        animationDuration={200}
      />

      <div className="flex justify-center gap-3">
        <button
          onClick={newGame}
          className="rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background hover:opacity-85"
        >
          New game
        </button>
        <button
          onClick={() => setOrientation((o) => (o === "white" ? "black" : "white"))}
          className="rounded-md border border-foreground/20 px-4 py-2 text-sm font-medium hover:bg-foreground/10"
        >
          Flip board
        </button>
      </div>
    </div>
  );
}

function gameStatus(game: Chess): string {
  const side = game.turn() === "w" ? "White" : "Black";
  const other = game.turn() === "w" ? "Black" : "White";
  if (game.isCheckmate()) return `Checkmate: ${other} wins`;
  if (game.isStalemate()) return "Draw by stalemate";
  if (game.isThreefoldRepetition()) return "Draw by threefold repetition";
  if (game.isInsufficientMaterial()) return "Draw by insufficient material";
  if (game.isDrawByFiftyMoves()) return "Draw by the 50-move rule";
  return game.isCheck() ? `${side} to move (check)` : `${side} to move`;
}
