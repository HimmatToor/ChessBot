"use client";

import { useRef, useState, type CSSProperties } from "react";
import { Chess, type Move, type Square } from "chess.js";
import { Chessboard } from "react-chessboard";
import { ArrowDownUp, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import GameOverDialog from "@/components/GameOverDialog";
import MoveList from "@/components/MoveList";
import PlayerBar from "@/components/PlayerBar";
import { gameResult, type GameResult } from "@/lib/gameResult";
import { cn } from "@/lib/utils";
import { materialBalance } from "@/lib/material";
import { playSound, type SoundName } from "@/lib/sounds";

type MoveSquares = { from: Square; to: Square };

// Fixed board theme: the same in light and dark mode.
// (react-chessboard types these as plain records, not CSSProperties.)
const LIGHT_SQUARE_STYLE = { backgroundColor: "#ebecd0" };
const DARK_SQUARE_STYLE = { backgroundColor: "#779556" };
const BOARD_STYLE = {
  borderRadius: "8px",
  boxShadow: "0 10px 30px rgba(0, 0, 0, 0.35)",
};

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
  const [dialogOpen, setDialogOpen] = useState(false);
  const dialogTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const legalMovesFrom = (square: Square) => game.moves({ square, verbose: true });

  const isPromotion = (from: Square, to: Square) =>
    legalMovesFrom(from).some((m) => m.to === to && m.promotion);

  function makeMove(from: Square, to: Square, promotion?: string): boolean {
    let move: Move;
    try {
      move = game.move({ from, to, promotion });
    } catch {
      return false; // chess.js throws on illegal moves
    }
    playSound(moveSound(game, move));
    setFen(game.fen());
    setLastMove({ from, to });
    setSelected(null);
    if (game.isGameOver()) {
      // Let the final move animate before the dialog covers the board.
      dialogTimer.current = setTimeout(() => setDialogOpen(true), 600);
    }
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
    if (dialogTimer.current) clearTimeout(dialogTimer.current);
    setDialogOpen(false);
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

  // The side at the bottom of the board is whichever colour it's oriented to.
  const bottomColor = orientation === "white" ? "w" : "b";
  const topColor = bottomColor === "w" ? "b" : "w";
  const gameOver = game.isGameOver();
  const result = gameResult(game);
  const status = statusText(game, result);
  const material = materialBalance(game.board());
  const playerBar = (color: "w" | "b") => (
    <PlayerBar
      name={color === "w" ? "White" : "Black"}
      color={color}
      isTurn={!gameOver && game.turn() === color}
      material={material[color]}
    />
  );

  return (
    <div className="grid w-full max-w-6xl gap-6 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-start">
      {/* Board column: capped by viewport height so the whole board fits on screen */}
      <section className="mx-auto flex w-full max-w-[min(680px,calc(100dvh_-_13rem))] min-w-[280px] flex-col gap-2">
        {playerBar(topColor)}
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
          isDraggablePiece={({ piece }) => !gameOver && piece[0] === game.turn()}
          customSquareStyles={squareStyles}
          customLightSquareStyle={LIGHT_SQUARE_STYLE}
          customDarkSquareStyle={DARK_SQUARE_STYLE}
          customBoardStyle={BOARD_STYLE}
          animationDuration={200}
        />
        {playerBar(bottomColor)}
      </section>

      {/* Side panel */}
      <aside className="flex flex-col gap-4 rounded-xl border bg-card p-4 text-card-foreground shadow-sm">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Status
          </p>
          {/* Keyed on the text so each change replays the fade-in. */}
          <p
            key={status}
            className={cn(
              "mt-1 text-lg font-semibold animate-in fade-in slide-in-from-bottom-1 duration-300",
              result && "text-primary",
            )}
            aria-live="polite"
          >
            {status}
          </p>
        </div>

        <MoveList moves={game.history()} />

        <div className="grid grid-cols-2 gap-2">
          <Button onClick={newGame}>
            <RefreshCw />
            New game
          </Button>
          <Button
            variant="outline"
            onClick={() => setOrientation((o) => (o === "white" ? "black" : "white"))}
          >
            <ArrowDownUp />
            Flip board
          </Button>
        </div>
      </aside>

      <GameOverDialog
        result={result}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onNewGame={newGame}
      />
    </div>
  );
}

function moveSound(game: Chess, move: Move): SoundName {
  if (game.isGameOver()) return "end";
  if (game.isCheck()) return "check";
  if (move.isCapture()) return "capture";
  return "move";
}

function statusText(game: Chess, result: GameResult | null): string {
  if (result) return `${result.title}: ${result.detail}`;
  const side = game.turn() === "w" ? "White" : "Black";
  return game.isCheck() ? `${side} to move (check)` : `${side} to move`;
}
