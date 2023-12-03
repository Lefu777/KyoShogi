#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"

#include "mate.h"

#include "cshogi.h"
#include "config.hpp"

const constexpr size_t MaxCheckMoves = 91;

// 詰み探索用のMovePicker
template <bool or_node, bool INCHECK>
class MovePicker {
public:
	explicit MovePicker(const Position& pos) {
		if (or_node) {
			last_ = generateMoves<CheckAll>(moveList_, pos);
			if (INCHECK) {
				// 自玉が王手の場合、逃げる手かつ王手をかける手を生成
				ExtMove* curr = moveList_;
				//const Bitboard pinned = pos.pinnedBB();
				while (curr != last_) {    // NOTE: 末尾に達するまで続ける。全部を走査。
					//if (!pos.pseudoLegalMoveIsEvasion(curr->move, pinned)) {
					if (!pos.moveIsPseudoLegal<false>(curr->move)) {
						// NOTE
						//     : 非合法手を発見した場合は、末尾から要素を持って来て上書きする。
						//       この時end -= 1 され、上書きした奴はこの次で検証される。
						curr->move = (--last_)->move;
					}
					else {
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
	size_t size() const { return static_cast<size_t>(last_ - moveList_); }
	ExtMove* begin() { return &moveList_[0]; }
	ExtMove* end() { return last_; }
	bool empty() const { return size() == 0; }

private:
	ExtMove moveList_[MaxCheckMoves];
	ExtMove* last_;
};

// 3手詰めチェック
// 手番側が王手でないこと
template <bool INCHECK>
FORCE_INLINE bool mateMoveIn3Ply(Position& pos)
{
	// OR節点

	StateInfo si;
	StateInfo si2;

	const CheckInfo ci(pos);
	for (const auto& ml : MovePicker<true, INCHECK>(pos))
	{
		const Move& m = ml.move;

		pos.doMove(m, si, ci, true);

		// この局面ですべてのevasionを試す
		MovePicker<false, false> move_picker2(pos);

		if (move_picker2.size() == 0) {
			// 1手で詰んだ
			pos.undoMove(m);
			return true;
		}

		const CheckInfo ci2(pos);
		for (const auto& move : move_picker2)
		{
			const Move& m2 = move.move;

			// この指し手で逆王手になるなら、不詰めとして扱う
			if (pos.moveGivesCheck(m2, ci2))
				goto NEXT_CHECK;

			pos.doMove(m2, si2, ci2, false);

			if (!pos.mateMoveIn1Ply<false>()) {
				// 詰んでないので、m2で詰みを逃れている。
				pos.undoMove(m2);
				goto NEXT_CHECK;
			}

			pos.undoMove(m2);
		}

		// すべて詰んだ
		pos.undoMove(m);
		return true;

	NEXT_CHECK:;
		pos.undoMove(m);
	}
	return false;
}

// 奇数手詰めチェック
// 詰ます手を返すバージョン
template <bool INCHECK>
Move mateMoveInOddPlyReturnMove(Position& pos, const int depth) {
	// NOTE: 一つでも詰ませる手が有れば良い。
	// OR節点

	// NOTE
	//     : MovePicker<true, INCHECK>(pos) にも書いてあるが、
	//       手番側に王手が掛けられている状態で、相手を詰ませるには、
	//       王手を逃れて、且つ王手を掛ける手のみ有効。
	//       (ex> 王が横によけることで、香車の利きを相手玉に通す。)
	//       (ex> 飛び駒の王手を防いで、その防いだ駒で逆に相手玉に王手を掛ける。)
	// すべての合法手について
	const CheckInfo ci(pos);

	// debug
	//const std::string sfen = pos.toSFEN();

	for (const auto& ml : MovePicker<true, INCHECK>(pos)) {
		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move, state, ci, true);

		// 千日手チェック
		switch (pos.isDraw(16)) {
		case NotRepetition: break;
		case RepetitionLose: // 相手が負け
		{
			// 詰みが見つかった時点で終了
			pos.undoMove(ml.move);
			return ml.move;
		}
		case RepetitionDraw:
		case RepetitionWin: // 相手が勝ち
		case RepetitionSuperior: // 相手が駒得
		{
			pos.undoMove(ml.move);
			continue;
		}
		case RepetitionInferior: break; // 相手が駒損
		default: UNREACHABLE;
		}

		//std::cout << ml.move().toUSI() << std::endl;
		// 偶数手詰めチェック
		if (mateMoveInEvenPly(pos, depth - 1)) {    // 偶数手詰めで詰み発見(相手が偶数手で詰まされることに気付く)
			// NOTE: この指し手が、depth手詰めの最初の手である。
			// 詰みが見つかった時点で終了

			pos.undoMove(ml.move);
			return ml.move;
		}

		pos.undoMove(ml.move);
	}
	return Move::moveNone();
}
template Move mateMoveInOddPlyReturnMove<true>(Position& pos, const int depth);
template Move mateMoveInOddPlyReturnMove<false>(Position& pos, const int depth);

// 奇数手詰めチェック
template <bool INCHECK>
bool mateMoveInOddPly(Position& pos, const int depth)
{
	// OR節点

	// すべての合法手について
	const CheckInfo ci(pos);
	for (const auto& ml : MovePicker<true, INCHECK>(pos)) {
		//std::cout << depth << " : " << pos.toSFEN() << " : " << ml.move.toUSI() << std::endl;
		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move, state, ci, true);

		// 千日手チェック
		switch (pos.isDraw(16)) {
		case NotRepetition: break;
		case RepetitionLose: // 相手が負け
		{
			// 詰みが見つかった時点で終了
			pos.undoMove(ml.move);
			return true;
		}
		case RepetitionDraw:
		case RepetitionWin: // 相手の勝ち
		case RepetitionSuperior: // 相手が駒得
		{
			pos.undoMove(ml.move);
			continue;
		}
		case RepetitionInferior: break; // 相手が駒損
		default: UNREACHABLE;
		}

		// 王手の場合
		// 偶数手詰めチェック
		if (mateMoveInEvenPly(pos, depth - 1)) {
			// 詰みが見つかった時点で終了
			pos.undoMove(ml.move);
			return true;
		}

		pos.undoMove(ml.move);
	}
	return false;
}

// NOTE
//     : 手番側が詰んでいるか否かを返す
//     : cython で以下のように用いられてい事と、以下のコメントから、
//       手番側が王手されている時(inCheck() == true) にのみ使用する。 
//       (連続王手で詰ませない限りは、あくまで必至？)
//       def is_mate(self, int ply):
//           assert ply % 2 == 0
//           return self.__board.inCheck() and self.__board.is_mate(ply)
// 偶数手詰めチェック
// 手番側が王手されていること
bool mateMoveInEvenPly(Position& pos, const int depth)
{
	// NOTE: どの手を指し手も、手番側が詰まされてしまう。(手番側「もう逃げられない...(｡＞﹏＜)」)
	// AND節点

	// debug
	//const std::string sfen = pos.toSFEN();


	// すべてのEvasionについて
	const CheckInfo ci(pos);
	for (const auto& ml : MovePicker<false, false>(pos)) {
		//std::cout << depth << " : " << pos.toSFEN() << " : " << ml.move.toUSI() << std::endl;
		const bool givesCheck = pos.moveGivesCheck(ml.move, ci);

		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move, state, ci, givesCheck);

		// 千日手チェック
		switch (pos.isDraw(16)) {
		case NotRepetition: break;
		case RepetitionWin: // 自分が勝ち
		{
			pos.undoMove(ml.move);
			continue;
		}
		case RepetitionDraw:
		case RepetitionLose: // 自分が負け
		case RepetitionInferior: // 自分が駒損
		{
			// 詰みが見つからなかった時点で終了
			pos.undoMove(ml.move);
			return false;
		}
		case RepetitionSuperior: break; // 自分が駒得
		default: UNREACHABLE;
		}

		if (depth == 4) {
			// 3手詰めかどうか
			if (givesCheck ? !mateMoveIn3Ply<true>(pos) : !mateMoveIn3Ply<false>(pos)) {
				// 3手詰めでない場合
				// 詰みが見つからなかった時点で終了
				pos.undoMove(ml.move);
				return false;
			}
		}
		else if (depth == 2) {
			// TODO
			//     : ここ、inCheck() 抜けてない？
			//       inCheck() でないにしろ、dfpn 内であったように、後手のあるEvasion(ml.move) が相手(OrNode)に王手を掛けるなら、
			//       この時pos はinCheck() なのでmateMoveIn1Ply() (を実行できないので)せずに、不詰みとみなす、的な事をすべきな気がする。自信はない。
			//     : 王手掛かってるとmateMoveIn1Ply() が上手く動かないのは(というか、assert で王手が掛かってる場合を弾いているのは)、
			//       これが簡易判定器だからかね？よー分からん。
			// 以下4行ぐらいは私が追加。
			if (givesCheck) {
				pos.undoMove(ml.move);
				return false;
			}

			// DEBUG: 一旦dlshogi に合わせてAdditional=false にしてる。(後もう一か所もそう。)
			// 1手詰めかどうか
			if (!pos.mateMoveIn1Ply<false>()) {    // 敵の手番で、1手で詰ませられない
				// 1手詰めでない場合
				// 詰みが見つからなかった時点で終了
				pos.undoMove(ml.move);
				return false;
			}
		}
		else {
			// 奇数手詰めかどうか
			if (givesCheck ? !mateMoveInOddPly<true>(pos, depth - 1) : !mateMoveInOddPly<false>(pos, depth - 1)) {
				// 偶数手詰めでない場合
				// 詰みが見つからなかった時点で終了
				pos.undoMove(ml.move);
				return false;
			}
		}

		pos.undoMove(ml.move);
	}
	return true;
}
