#pragma once
#include "cshogi.h"

#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"


#include "cshogi.h"
#include "util.hpp"
#include "config.hpp"

const constexpr size_t MaxCheckMoves = 91;

// 詰み探索用のMovePicker
template <bool or_node, bool INCHECK>
class MovePickerTest {
public:
	explicit MovePickerTest(const Position& pos) {
		std::cout << "info: MovePickerTest constructor" << std::endl;
		if (or_node) {
			last_ = generateMoves<CheckAll>(moveList_, pos);
			if (INCHECK) {
				// 自玉が王手の場合、逃げる手かつ王手をかける手を生成
				ExtMove* curr = moveList_;
				const Bitboard pinned = pos.pinnedBB();
				while (curr != last_) {    // NOTE: 末尾に達するまで続ける。全部を走査。
					if (!pos.pseudoLegalMoveIsEvasion(curr->move, pinned)) {
						std::cout
							<< "info: MovePickerTest delete move"
							<< ",move=" << curr->move.value()
							<< ",moveUSI=" << curr->move.toUSI()
							<< std::endl;
						// NOTE
						//     : 非合法手を発見した場合は、末尾から要素を持って来て上書きする。
						//       この時end -= 1 され、上書きした奴はこの次で検証される。
						curr->move = (--last_)->move;
					}
					else {
						std::cout
							<< "info: MovePickerTest isOk"
							<< ",move=" << curr->move.value()
							<< ",moveUSI=" << curr->move.toUSI()
							<< std::endl;
						++curr;
					}
				}
			}
		}
		else {
			last_ = generateMoves<Evasion>(moveList_, pos);
			// 玉の移動による自殺手と、pinされている駒の移動による自殺手を削除
			ExtMove* curr = moveList_;
			const Bitboard pinned = pos.pinnedBB();
			while (curr != last_) {
				if (!pos.pseudoLegalMoveIsLegal<false, false>(curr->move, pinned))
					curr->move = (--last_)->move;
				else
					++curr;
			}
		}
		assert(size() <= MaxCheckMoves);
	}
	size_t size() const {
		std::cout << "info: MovePickerTest size()" << std::endl;
		return static_cast<size_t>(last_ - moveList_);
	}
	ExtMove* begin() { return &moveList_[0]; }
	ExtMove* end() { return last_; }
	bool empty() const { return size() == 0; }

private:
	ExtMove moveList_[MaxCheckMoves];
	ExtMove* last_;
};

#ifdef CSHOGI_BOARD_MATE_MOVE_TEST

inline void print_bb(const Bitboard& bb) {
	for (Square sq = SQ11; sq < SquareNum; ++sq) {
		std::cout << bb.isSet(sq) << " ";
		if (sq % 9 == 8) {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

void cshogi_board_mate_move_test0() {
	const std::string sfen = "lr6l/4g4/p3p4/1pp5p/P2S1p3/2P1N2+bP/1PGP1Pk1R/2N3pp1/4K3L w b2g3s2nl4p 152";
	__Board board;
	board.set(sfen);
	//const auto mateMove = board.mateMove(3);
	//std::cout << "info mateMove = " << Move(mateMove).toUSI() << std::endl;

	const bool INCHECK = true;
	MovePickerTest<true, INCHECK> move_picker(board.pos);

	const auto attacks_from_rook = board.pos.attacksFrom<Rook  >(SQ27);
	print_bb(attacks_from_rook);

	int i = 0;
	std::cout << "info: " << "size=" << move_picker.size() << std::endl;
	for (const auto& ml : move_picker) {
		std::cout << "info: " << "i=" << i << ",move=" << ml.move.toUSI() << std::endl;
		std::cout << "    " << "to=" << ml.move.to() << std::endl;
		std::cout << "    " << "oppositeColor=" << oppositeColor(board.pos.turn()) << std::endl;
		std::cout << "    " << "(oppositeColor==Black)=" << bts(oppositeColor(board.pos.turn()) == Black) << std::endl;
		const bool isAny = board.pos.attackersToIsAny(oppositeColor(board.pos.turn()), ml.move.to());
		std::cout << "    " << "isAny=" << bts(isAny) << std::endl;

		const auto attacks_from_rook = board.pos.attacksFrom<Rook  >(ml.move.to());
		print_bb(attacks_from_rook);

		// 盤上にある飛車, 竜のbb
		const auto bbof_rook_dragon = board.pos.bbOf(Rook, Dragon);
		print_bb(bbof_rook_dragon);

		board.push(ml.move);

		std::cout << board.dump() << std::endl;
		//const auto occupied_bb_child = board.pos.occupiedBB();

		board.pop();

		++i;
	}

	const auto occupied_bb = board.pos.occupiedBB();
	print_bb(occupied_bb);

	const auto pinned_bb = board.pos.pinnedBB();
	print_bb(pinned_bb);


	//FOREACH_BB(occupied_bb, Square sq, {
	//	const Piece pc = board.pos.piece(sq);
	//	const PieceType pt = pieceToPieceType(pc);
	//	Color c = pieceToColor(pc);
	//	});

}

#endif


// test