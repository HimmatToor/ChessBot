import io
import zipfile

import chess
import chess.pgn
import numpy as np

import prepare_data
from chessnet.data import DRAW, LOSS, WIN, ShardWriter, game_examples, load_shard, pack_planes, unpack_planes
from chessnet.encode import MOVES, encode_board

SCHOLARS_MATE = """[Event "Rated Blitz game"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "2400"]
[BlackElo "2300"]
[TimeControl "180+0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""


def make_pgn(result: str = "1-0", white_elo: int = 2400, black_elo: int = 2300,
             time_control: str = "180+0", moves: str = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6") -> str:
    return (f'[Result "{result}"]\n[WhiteElo "{white_elo}"]\n[BlackElo "{black_elo}"]\n'
            f'[TimeControl "{time_control}"]\n\n{moves} {result}\n\n')


def read(pgn: str) -> chess.pgn.Game:
    return chess.pgn.read_game(io.StringIO(pgn))


def test_pack_round_trip():
    planes = np.stack([encode_board(chess.Board()), encode_board(chess.Board("8/8/4k3/8/8/3K4/8/8 b - - 0 1"))])
    packed = pack_planes(planes)
    assert packed.shape == (2, 144)
    np.testing.assert_array_equal(unpack_planes(packed), planes)


def test_game_examples_results_are_from_side_to_move():
    examples = list(game_examples(read(SCHOLARS_MATE)))
    assert len(examples) == 7
    # White won: positions with White to move are WIN, Black to move are LOSS.
    assert [result for _, _, result in examples] == [WIN, LOSS, WIN, LOSS, WIN, LOSS, WIN]
    planes, move, _ = examples[0]
    np.testing.assert_array_equal(planes, encode_board(chess.Board()))
    assert MOVES[move] == "e2e4"
    # Black's 1...e5 is stored in Black's mirrored frame.
    assert MOVES[examples[1][1]] == "e2e4"


def test_game_examples_draw_and_skip():
    examples = list(game_examples(read(make_pgn(result="1/2-1/2")), skip_plies=2))
    assert len(examples) == 4
    assert all(result == DRAW for _, _, result in examples)


def test_unfinished_games_give_no_examples():
    assert list(game_examples(read(make_pgn(result="*")))) == []


def test_shard_writer_splits_and_round_trips(tmp_path):
    writer = ShardWriter(tmp_path, "train", shard_size=3)
    for planes, move, result in game_examples(read(SCHOLARS_MATE)):
        writer.add(planes, move, result)
    writer.flush()
    assert writer.total == 7 and writer.shards_written == 3  # 3 + 3 + 1
    shard = load_shard(tmp_path / "train-0000.npz")
    assert shard["planes"].shape == (3, 144) and shard["move"].dtype == np.int16
    np.testing.assert_array_equal(unpack_planes(shard["planes"])[0], encode_board(chess.Board()))


def test_accept_game_filters():
    accept = lambda pgn: prepare_data.accept_game(read(pgn).headers, min_elo=2000, min_seconds=180)
    assert accept(make_pgn())
    assert not accept(make_pgn(black_elo=1900))  # too weak
    assert not accept(make_pgn(time_control="60+0"))  # bullet
    assert accept(make_pgn(time_control="120+2"))  # 120 + 80 = 200 s
    assert not accept(make_pgn(time_control="-"))  # correspondence / unknown
    assert not accept(make_pgn(result="*"))


def test_cli_end_to_end_from_zip(tmp_path):
    games = [make_pgn() for _ in range(4)] + [make_pgn(black_elo=1500)] + [SCHOLARS_MATE]
    archive = tmp_path / "games.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("games.pgn", "\n".join(games))

    out = tmp_path / "processed"
    prepare_data.main([str(archive), "--out", str(out), "--skip-plies", "0", "--val-every", "5"])

    train = load_shard(out / "train-0000.npz")
    val = load_shard(out / "val-0000.npz")
    # 5 accepted games: #0 is validation (6 plies); #1-3 (6 each) + Scholar's mate (7) are train.
    assert len(val["move"]) == 6
    assert len(train["move"]) == 3 * 6 + 7
