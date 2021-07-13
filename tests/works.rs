use std::io::Cursor;

use binread::BinRead;
use nnue::*;
use nnue::stockfish::halfkp::{SfHalfKpFullModel, SfHalfKpModel};

fn activate(model: &SfHalfKpModel, fen: &str) -> i32 {
    let mut parts = fen.split_ascii_whitespace();
    let pos = parts.next().unwrap();
    let mut pieces = Vec::new();
    for (rank, row) in pos.rsplit("/").enumerate() {
        let mut file = 0;
        for p in row.chars() {
            if let Some(offset) = p.to_digit(10) {
                file += offset as usize;
            } else {
                let piece = match p.to_ascii_lowercase() {
                    'p' => Piece::Pawn,
                    'n' => Piece::Knight,
                    'b' => Piece::Bishop,
                    'r' => Piece::Rook,
                    'q' => Piece::Queen,
                    'k' => Piece::King,
                    _ => panic!("Invalid piece {}", p)
                };
                let color = if p.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                let square = Square::from_index(rank * 8 + file);
                pieces.push((piece, color, square));
                file += 1;
            }
        }
    }
    let side_to_move = if parts.next().unwrap() == "w" {
        Color::White
    } else {
        Color::Black
    };
    let mut white_king = Square::A1;
    let mut black_king = Square::A1;
    for &(piece, color, square) in &pieces {
        if piece == Piece::King {
            if color == Color::White {
                white_king = square;
            } else {
                black_king = square;
            }
        }
    }
    let mut state = model.new_state(white_king, black_king);
    for &(piece, piece_color, square) in &pieces {
        if piece != Piece::King {
            for &color in &Color::ALL {
                state.add(color, piece, piece_color, square);
            }
        }
    }
    state.activate(side_to_move)[0] / 16
}

const FENS: &[(&'static str, i32)] = &[
    ("r1bq1rk1/p2nppbp/2p3pB/1p2P3/2pP4/2P2N2/P2QBPPP/R4RK1 b - - 7 12", -97),
    ("7k/6b1/4Q2p/7P/8/8/5PP1/6K1 w - - 5 42", 3146),
    ("8/5pk1/2R4p/6pP/6P1/pBP5/Pr5r/5K2 w - - 2 48", -2134),
    ("r2q1rk1/pp1n1ppp/2pb1p2/5b2/3P4/2P2N2/PP2BPPP/R1BQ1RK1 w - - 5 10", -2),
    ("1Q6/5pkp/4q1p1/6Pr/8/P5RP/5P2/6K1 w - - 2 44", 11),
    ("3qr1k1/R4p1p/p1r3p1/3pP3/3Bb3/P5P1/5P1P/3QR1K1 b - - 0 29", -54),
    ("8/5p1k/6p1/8/1p5p/1P5P/4QPPK/2q5 b - - 7 55", 0),
    ("r1bqk2r/ppp1ppbp/1n4p1/n2P4/4P3/2N1BP2/PP4PP/R2QKBNR w KQkq - 1 9", 256),
    ("5rk1/p1p1pp1p/N5p1/8/8/6Pb/PR2KPnP/7R b - - 4 23", -257),
    ("5rk1/4Rbq1/3r2pp/1p1p4/p2P4/2PNR2P/4QPP1/6K1 b - - 1 48", -610)
];

#[test]
fn nnue_impl_matches_up() {
    //Network grabbed from https://tests.stockfishchess.org/api/nn/nn-62ef826d1a6d.nnue
    let mut reader = Cursor::new(std::fs::read("tests/nn.nnue").unwrap());
    let model = SfHalfKpFullModel::read(&mut reader).unwrap();
    assert_eq!(model.desc, "Features=HalfKP(Friend)[41024->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))");
    for &(fen, eval) in FENS {
        assert_eq!(activate(&model.model, fen), eval);
    }
}
