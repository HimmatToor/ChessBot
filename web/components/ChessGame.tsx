"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Chess, type Color, type Move, type Square } from "chess.js";
import { Chessboard } from "react-chessboard";
import { ArrowDownUp, Check, Copy, Download, RefreshCw, Undo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import GameOverDialog from "@/components/GameOverDialog";
import MoveList from "@/components/MoveList";
import PlayerBar from "@/components/PlayerBar";
import { randomBot, type Bot, type BotMove } from "@/lib/bot";
import { DEFAULT_LEVEL, LEVELS, levelById, type LevelId } from "@/lib/engine/levels";
import { createEngine } from "@/lib/engine/neuralBot";
import SegmentedControl from "@/components/SegmentedControl";
import { gameResult, type GameResult } from "@/lib/gameResult";
import { exportPgn } from "@/lib/pgn";
import { cn } from "@/lib/utils";
import { materialBalance } from "@/lib/material";
import { playSound, type SoundName } from "@/lib/sounds";

type MoveSquares = { from: Square; to: Square };

// One engine (one worker, one model download) shared by a bot per level.
const engine = createEngine();
const BOTS = Object.fromEntries(
  LEVELS.map((l) => [l.id, engine.bot(l.settings, `ChessBot · ${l.label}`)]),
) as Record<LevelId, Bot>;
// Even an instant bot waits a moment, so its reply doesn't feel jarring.
const BOT_MIN_DELAY_MS = 400;
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

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
  const [playerColor, setPlayerColor] = useState<Color>("w");
  const [level, setLevel] = useState<LevelId>(DEFAULT_LEVEL);
  const bot = BOTS[level];
  const [botThinking, setBotThinking] = useState(false);
  const [botError, setBotError] = useState<string | null>(null);
  // Bumped whenever the position is reset or taken back, so a bot reply
  // that was still being computed for the old position gets discarded.
  const botToken = useRef(0);
  const [dialogOpen, setDialogOpen] = useState(false);
  const dialogTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");

  // Start downloading the model as soon as the page loads, not on the first bot move.
  useEffect(() => {
    engine.prepare().catch(() => {}); // a failure resurfaces (and is handled) on the bot's move
  }, []);

  const legalMovesFrom = (square: Square) => game.moves({ square, verbose: true });

  const isPromotion = (from: Square, to: Square) =>
    legalMovesFrom(from).some((m) => m.to === to && m.promotion);

  const isPlayersTurn = () => !game.isGameOver() && !botThinking && game.turn() === playerColor;

  /** Plays a move for either side and updates the UI. */
  function applyMove(input: BotMove): Move | null {
    let move: Move;
    try {
      move = game.move(input);
    } catch {
      return null; // chess.js throws on illegal moves
    }
    playSound(moveSound(game, move));
    setFen(game.fen());
    setLastMove({ from: move.from, to: move.to });
    setSelected(null);
    if (game.isGameOver()) {
      // Let the final move animate before the dialog covers the board.
      dialogTimer.current = setTimeout(() => setDialogOpen(true), 600);
    }
    return move;
  }

  /** Human move from a drag, click or promotion pick. */
  function makeMove(from: Square, to: Square, promotion?: string): boolean {
    if (!isPlayersTurn()) return false;
    const move = applyMove({ from, to, promotion });
    if (move && !game.isGameOver()) void playBotMove();
    return move !== null;
  }

  // `activeBot` defaults to the current level; newGame passes the new one
  // explicitly because state updates haven't been applied yet at that point.
  async function playBotMove(activeBot: Bot = bot) {
    const token = ++botToken.current;
    setBotThinking(true);
    let move: BotMove;
    try {
      [move] = await Promise.all([activeBot.chooseMove(game), delay(BOT_MIN_DELAY_MS)]);
    } catch (error) {
      // e.g. the model failed to download: keep the game going rather than freezing.
      console.error("ChessBot engine failed; falling back to random moves", error);
      setBotError("The ChessBot engine couldn't load, so it's playing random moves.");
      move = await randomBot.chooseMove(game);
    }
    if (token !== botToken.current) return; // position changed while thinking
    setBotThinking(false);
    applyMove(move);
  }

  function cancelBot() {
    botToken.current++;
    setBotThinking(false);
  }

  function closeDialog() {
    if (dialogTimer.current) clearTimeout(dialogTimer.current);
    setDialogOpen(false);
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
    if (!isPlayersTurn()) return;

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

  /** Takes back your last move (and the bot's reply, if it has made one). */
  function undo() {
    cancelBot();
    closeDialog();
    game.undo();
    if (game.turn() !== playerColor) game.undo();
    const previous = game.history({ verbose: true }).at(-1);
    setLastMove(previous ? { from: previous.from, to: previous.to } : null);
    setFen(game.fen());
    setSelected(null);
    setPendingPromotion(null);
  }

  const pgn = () =>
    exportPgn(game, {
      white: playerColor === "w" ? "You" : bot.name,
      black: playerColor === "b" ? "You" : bot.name,
    });

  async function copyPgn() {
    try {
      await navigator.clipboard.writeText(pgn());
      setCopyState("copied");
    } catch {
      setCopyState("failed"); // clipboard needs a secure context + permission
    }
    setTimeout(() => setCopyState("idle"), 2000);
  }

  function downloadPgn() {
    const blob = new Blob([pgn()], { type: "application/x-chess-pgn" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `chessbot-${new Date().toISOString().slice(0, 10)}.pgn`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function newGame(color: Color = playerColor, newLevel: LevelId = level) {
    cancelBot();
    closeDialog();
    game.reset();
    setFen(game.fen());
    setSelected(null);
    setLastMove(null);
    setPendingPromotion(null);
    setPlayerColor(color);
    setLevel(newLevel);
    setOrientation(color === "w" ? "white" : "black");
    if (color === "b") void playBotMove(BOTS[newLevel]); // bot opens as White
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
  // Not bot.name: the level is already shown in the player bar.
  const status = botThinking ? "ChessBot is thinking…" : statusText(game, result);
  const moves = game.history();
  const hasMoves = moves.length > 0;
  // Undo needs at least one of *your* moves (as Black, the first move is the bot's).
  const canUndo = moves.length >= (playerColor === "w" ? 1 : 2);
  const material = materialBalance(game.board());
  const playerBar = (color: Color) => (
    <PlayerBar
      name={color === playerColor ? "You" : bot.name}
      color={color}
      isTurn={!gameOver && game.turn() === color}
      material={material[color]}
    />
  );

  return (
    // Board and panel sit side by side as one centred unit; the panel stretches to the board's height.
    <div className="flex w-full flex-col items-center gap-4 lg:flex-row lg:items-stretch lg:justify-center">
      {/* Board column: exactly the board's width. The board is as big as the window allows:
          its height minus header, padding and the two player bars (~12rem), and on wide
          layouts its width minus the side panel (~24rem). */}
      <section className="flex w-full max-w-[min(760px,calc(100dvh_-_12rem))] min-w-[280px] flex-col gap-1.5 lg:w-[min(760px,calc(100dvh_-_12rem),calc(100vw_-_24rem))] lg:max-w-none lg:shrink-0">
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
          isDraggablePiece={({ piece }) => isPlayersTurn() && piece[0] === playerColor}
          customSquareStyles={squareStyles}
          customLightSquareStyle={LIGHT_SQUARE_STYLE}
          customDarkSquareStyle={DARK_SQUARE_STYLE}
          customBoardStyle={BOARD_STYLE}
          animationDuration={200}
        />
        {playerBar(bottomColor)}
      </section>

      {/* Side panel */}
      <aside className="flex w-full max-w-[min(760px,calc(100dvh_-_12rem))] min-w-[280px] flex-col gap-3 rounded-xl border bg-card p-4 text-card-foreground shadow-sm lg:w-80 lg:max-w-none lg:shrink-0">
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
          {botError && <p className="mt-1 text-sm text-destructive">{botError}</p>}
        </div>

        {/* Grows to fill whatever height the board leaves free in the panel. */}
        <MoveList moves={moves} className="h-40 lg:h-auto lg:min-h-32 lg:flex-1" />

        <div className="flex flex-col gap-3 border-t pt-3">
          <SegmentedControl
            label="Play as"
            value={playerColor}
            onChange={(color) => newGame(color, level)}
            options={[
              { value: "w", label: "White", title: "Start a new game as White" },
              { value: "b", label: "Black", title: "Start a new game as Black" },
            ]}
          />
          <SegmentedControl
            label="Level"
            value={level}
            onChange={(id) => newGame(playerColor, id)}
            options={LEVELS.map((l) => ({ value: l.id, label: l.label, title: `Start a new ${l.label} game` }))}
          />
          <p className="text-xs leading-relaxed text-muted-foreground">
            {levelById(level).description}. Changing colour or level starts a new game.
          </p>
        </div>

        <div className="flex flex-col gap-2 border-t pt-3">
          <Button onClick={() => newGame()} className="w-full">
            <RefreshCw />
            New game
          </Button>
          {/* Two per row so every label fits, including "Copied" feedback. */}
          <div className="grid grid-cols-2 gap-2">
            <Button variant="outline" onClick={undo} disabled={!canUndo}>
              <Undo2 />
              Undo
            </Button>
            <Button
              variant="outline"
              onClick={() => setOrientation((o) => (o === "white" ? "black" : "white"))}
            >
              <ArrowDownUp />
              Flip board
            </Button>
            <Button variant="outline" onClick={copyPgn} disabled={!hasMoves}>
              {copyState === "copied" ? <Check /> : <Copy />}
              {copyState === "copied" ? "Copied!" : copyState === "failed" ? "Copy failed" : "Copy PGN"}
            </Button>
            <Button variant="outline" onClick={downloadPgn} disabled={!hasMoves}>
              <Download />
              Download
            </Button>
          </div>
        </div>
      </aside>

      <GameOverDialog
        result={result}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onNewGame={() => newGame()}
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
