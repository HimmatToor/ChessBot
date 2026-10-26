// PUCT Monte-Carlo tree search guided by the policy/value network
// (the AlphaZero / Leela Chess Zero algorithm).
//
// Value convention: every value is in [-1, 1]. A node's `valueSum` is from the
// point of view of the player who made the move leading to it, so a parent
// simply prefers children with high average value.

import type { Chess } from "chess.js";
import { encodeBoard, legalMoves, PLANES_SIZE, type LegalMove, type MoveInput } from "./encode";
import { popMove, pushMove, type InternalMove } from "./fastMoves";
import type { NetworkOutput } from "./network";
import { interpretLegal } from "./policy";

/** Runs the network on `batch` positions packed into `planes`. */
export type Evaluator = (planes: Float32Array, batch: number) => Promise<NetworkOutput>;

export type SearchOptions = {
  /** Stop after this many simulations (network evaluations + terminal hits). */
  simulations: number;
  /** Optional wall-clock budget; whichever limit is hit first ends the search. */
  timeLimitMs?: number;
  /** Positions evaluated per network call. */
  batchSize?: number;
  /** Exploration constant: higher trusts the policy prior more. */
  cPuct?: number;
};

export type SearchResult = {
  move: MoveInput;
  /** Root's expected score for the side to move, -1..1. */
  value: number;
  /** Root children, most visited first. */
  candidates: { move: MoveInput; visits: number; prior: number; q: number }[];
  /** Principal variation in UCI, starting with `move`. */
  pv: string[];
  simulations: number;
};

const DEFAULT_BATCH = 16;
const DEFAULT_C_PUCT = 1.5;
/** Unvisited moves are assumed a bit worse than their parent ("first play urgency"). */
const FPU_REDUCTION = 0.25;

class Node {
  visits = 0;
  valueSum = 0;
  /** null until expanded. Empty array = terminal position. */
  children: Node[] | null = null;
  /** Value of a terminal position for the side to move in it. */
  terminalValue: number | null = null;
  /** Network value for the side to move here (used for FPU). */
  netValue = 0;
  /** Queued for evaluation in the current batch. */
  pending = false;

  constructor(
    readonly move: MoveInput | null,
    readonly prior: number,
    /** chess.js internal move, for fast push/pop during the search. */
    readonly internal: InternalMove | null = null,
  ) {}

  /** Average value from the point of view of the player who chose this node. */
  get q(): number {
    return this.visits === 0 ? 0 : this.valueSum / this.visits;
  }
}

const uci = (m: MoveInput) => m.from + m.to + (m.promotion ?? "");

function select(parent: Node, cPuct: number): Node {
  const children = parent.children!;
  const sqrtVisits = Math.sqrt(parent.visits);
  // Parent's value from the chooser's (= side to move at parent) perspective.
  const fpu = parent.netValue - FPU_REDUCTION;
  let best = children[0];
  let bestScore = -Infinity;
  for (const child of children) {
    const q = child.visits === 0 ? fpu : child.q;
    const score = q + (cPuct * child.prior * sqrtVisits) / (1 + child.visits);
    if (score > bestScore) {
      bestScore = score;
      best = child;
    }
  }
  return best;
}

/** Value for the side to move if the game is over here, else null. */
function terminalValue(game: Chess, hasLegalMoves: boolean): number | null {
  if (!hasLegalMoves) return game.isCheck() ? -1 : 0; // checkmated, or stalemate
  if (game.isInsufficientMaterial() || game.isThreefoldRepetition() || game.isDrawByFiftyMoves()) return 0;
  return null;
}

/** Adds `value` (for the side to move at the leaf) to every node on the path. */
function backpropagate(path: Node[], value: number) {
  // The leaf's own valueSum is from the perspective of the player who moved into it,
  // i.e. the opponent of the side to move there.
  let v = -value;
  for (let i = path.length - 1; i >= 0; i--) {
    path[i].visits += 1;
    path[i].valueSum += v;
    v = -v;
  }
}

function expand(node: Node, legal: LegalMove[], output: NetworkOutput, row: number) {
  const evaluation = interpretLegal(legal, output.policy, output.wdl, row);
  node.netValue = evaluation.value;
  node.children = evaluation.moves.map(({ move, prior, internal }) => new Node(move, prior, internal));
  return evaluation.value;
}

/**
 * Searches from the current position of `game` and returns the best move.
 * `game` is temporarily modified during the search and restored afterwards.
 */
export async function search(game: Chess, evaluate: Evaluator, options: SearchOptions): Promise<SearchResult> {
  const batchSize = options.batchSize ?? DEFAULT_BATCH;
  const cPuct = options.cPuct ?? DEFAULT_C_PUCT;
  const deadline = options.timeLimitMs === undefined ? Infinity : performance.now() + options.timeLimitMs;
  const planes = new Float32Array(batchSize * PLANES_SIZE);

  const rootLegal = legalMoves(game);
  if (rootLegal.length === 0) throw new Error("no legal moves to search");
  const root = new Node(null, 1);
  encodeBoard(game, planes, 0);
  backpropagate([root], expand(root, rootLegal, await evaluate(planes, 1), 0));

  type Leaf = { path: Node[]; legal: LegalMove[] };

  while (root.visits < options.simulations && performance.now() < deadline) {
    const leaves: Leaf[] = [];

    while (leaves.length < batchSize && root.visits + leaves.length < options.simulations) {
      // 1. Select: walk down to a node that hasn't been evaluated yet.
      const path = [root];
      let node = root;
      while (node.children && node.children.length > 0) {
        node = select(node, cPuct);
        pushMove(game, node.internal!);
        path.push(node);
      }

      if (node.pending) {
        // Another leaf in this batch already took this path: evaluate what we have.
        for (let i = 1; i < path.length; i++) popMove(game);
        break;
      }

      // 2. Terminal positions are scored by the rules, without the network.
      const legal = node.children ? [] : legalMoves(game);
      const terminal = node.terminalValue ?? terminalValue(game, legal.length > 0);
      if (terminal !== null) {
        node.terminalValue = terminal;
        node.children = [];
        backpropagate(path, terminal);
      } else {
        // 3. Queue for the network, with a virtual loss so the rest of this
        //    batch explores elsewhere.
        encodeBoard(game, planes, leaves.length * PLANES_SIZE);
        node.pending = true;
        for (let i = 1; i < path.length; i++) {
          path[i].visits += 1;
          path[i].valueSum -= 1;
        }
        leaves.push({ path, legal });
      }
      for (let i = 1; i < path.length; i++) popMove(game);
    }

    if (leaves.length === 0) continue;

    // 4. Evaluate the batch, expand, undo the virtual loss and back up the values.
    const output = await evaluate(planes, leaves.length);
    leaves.forEach(({ path, legal }, row) => {
      const leaf = path[path.length - 1];
      leaf.pending = false;
      for (let i = 1; i < path.length; i++) {
        path[i].visits -= 1;
        path[i].valueSum += 1;
      }
      backpropagate(path, expand(leaf, legal, output, row));
    });
  }

  const children = [...root.children!].sort((a, b) => b.visits - a.visits || b.prior - a.prior);
  const pv: string[] = [];
  for (let node: Node | undefined = children[0]; node; ) {
    pv.push(uci(node.move!));
    node = node.children?.length
      ? node.children.reduce((a, b) => (b.visits > a.visits ? b : a))
      : undefined;
    if (node && node.visits === 0) break;
  }

  return {
    move: children[0].move!,
    value: -root.q, // root.valueSum is stored from the (non-existent) previous mover's view
    candidates: children.map((c) => ({ move: c.move!, visits: c.visits, prior: c.prior, q: c.q })),
    pv,
    simulations: root.visits,
  };
}
