#include <unordered_set>

#include "dfpn_dlshogi.hpp"

#include "config.hpp"
#ifdef DFPN_DLSHOGI

using namespace std;
using namespace ns_dfpn;


// NOTE
//     : REPEAT って、千日手 or 最大手数で中断 の場合にset されて、そいつらを他と区別する為？
int64_t DfPn::HASH_SIZE_MB = 2048;
int DfPn::draw_ply = INT_MAX;
const constexpr uint32_t REPEAT = UINT_MAX - 1;

// --- 詰み将棋探索

void DfPn::dfpn_stop(const bool stop)
{
	this->stop = stop;
}

// NOTE: まじで以下のコメントの通りで、詰将棋する上であり得る指し手を全て列挙する。
// 詰将棋エンジン用のMovePicker
namespace ns_dfpn {
	// NOTE: 任意の局面における王手を掛ける指し手の数は、MaxCheckMoves 以下である？
	const constexpr size_t MaxCheckMoves = 91;

	template <bool or_node>
	class MovePicker {
	public:
		explicit MovePicker(const Position& pos) {
			if (or_node) {
				// NOTE: この関数によって、moveList_ がbegin, last_ がend となる。
				last_ = generateMoves<CheckAll>(moveList_, pos);
				if (pos.inCheck()) {
					// 自玉が王手の場合、逃げる手かつ王手をかける手を生成
					ExtMove* curr = moveList_;
					while (curr != last_) {
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
				last_ = generateMoves<Evasion>(moveList_, pos);    // NOTE: コイツはpseudoLegal を返す
				// 玉の移動による自殺手と、pinされている駒の移動による自殺手を削除
				ExtMove* curr = moveList_;
				const Bitboard&& pinned = pos.pinnedBB();    // NOTE: 非参照から右辺値参照に変更
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
}

// 置換表
TranspositionTable::~TranspositionTable() {
	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}
}

TTEntry& TranspositionTable::LookUp(const Key key, const Hand hand, const uint16_t depth) {
	auto& entries = tt[key & clusters_mask];
	uint32_t hash_high = key >> 32;
	return LookUpDirect(entries, hash_high, hand, depth);
}

// NOTE
//     : 引数hand には、LookUp<bool or_node>() を通じてOR なら手番側の持ち駒が、AND には敵側の持ち駒が渡される(はず)。
//       -> 要するに、hand は詰ます側の持ち駒が渡される。
TTEntry& TranspositionTable::LookUpDirect(
	Cluster& entries, const uint32_t hash_high, const Hand hand, const uint16_t depth
) {
	int max_pn = 1;
	int max_dn = 1;
	// 検索条件に合致するエントリを返す
	for (size_t i = 0; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
		TTEntry& entry = entries.entries[i];

		// NOTE
		//     : 今の世代に一致しないものは、ゴミと見なす？(利用せずに上書きする？)
		if (generation != entry.generation) {
			// 空のエントリが見つかった場合
			entry.hash_high = hash_high;
			entry.depth = depth;
			entry.hand = hand;
			// TODO
			//     : 最悪の場合を想定してmax を登録しておく？
			//       だとしたら途中までじゃなくてなんで全体の最大値にしないの？
			entry.pn = max_pn;
			entry.dn = max_dn;
			entry.generation = generation;
			entry.num_searched = 0;
			return entry;
		}

		// TODO
		//     : hash と世代が一致するのは絶対に必要？ generation が進むタイミングは？(see dlshogi)
		if (hash_high == entry.hash_high && generation == entry.generation) {
			if (hand == entry.hand && depth == entry.depth) {
				// keyが合致するエントリを見つけた場合
				// 残りのエントリに優越関係を満たす局面があり証明済みの場合、それを返す
				for (i++; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
					TTEntry& entry_rest = entries.entries[i];
					if (generation != entry_rest.generation) break;
					if (hash_high == entry_rest.hash_high) {
						// NOTE
						//     : 詰ます側が、残りのentry の持ち駒以上に持ち駒を持っているなら上位互換で、詰むはず、的な話？
						//       -> key でアクセスしてるので、盤面だけで見ても一致するとは限らんよな？そのあたりどうなってん？
						if (entry_rest.pn == 0) {    // NOTE:  詰みが証明済み
							if (hand.isEqualOrSuperior(entry_rest.hand) && entry_rest.num_searched != REPEAT) {
								return entry_rest;
							}
						}
						else if (entry_rest.dn == 0) {    // NOTE: 不詰み証明済み
							if (entry_rest.hand.isEqualOrSuperior(hand) && entry_rest.num_searched != REPEAT) {
								return entry_rest;
							}
						}
					}
				}
				// NOTE: より強いことが言えるentry が無ければ、先に見つかった奴を返す
				return entry;
			}
			// TODO
			//     : 何故hash さえ一致して、持ち駒の優越関係を満たす証明済み局面があればそれを返す、にせずに、
			//       generation が一致することを確認してるの？
			//       -> L107 のif (generation != entry.generation) { からの節を見ても分かるように、
			//          前の古い世代の探索結果は用いず、今の世代の探索結果のみを用いるから。
			// 優越関係を満たす局面に証明済みの局面がある場合、それを返す
			if (entry.pn == 0) {
				if (hand.isEqualOrSuperior(entry.hand) && entry.num_searched != REPEAT) {
					return entry;
				}
			}
			else if (entry.dn == 0) {
				if (entry.hand.isEqualOrSuperior(hand) && entry.num_searched != REPEAT) {
					return entry;
				}
			}
			// TODO: 何故以下の場合に最大値を更新するのか？
			else if (entry.hand.isEqualOrSuperior(hand)) {
				if (entry.pn > max_pn) max_pn = entry.pn;
			}
			else if (hand.isEqualOrSuperior(entry.hand)) {
				if (entry.dn > max_dn) max_dn = entry.dn;
			}
		}
	}

	// NOTE
	//     : 探索数が一番少ない局面程、hash に貯めておく価値も低いので、探索数が一番少ないentry を見つけて潰す。
	//       その際に、詰みが証明できていない局面から優先的に潰すようにする。
	//cout << "hash entry full" << endl;
	// 合致するエントリが見つからなかったので
	// 古いエントリをつぶす
	TTEntry* best_entry = nullptr;
	uint32_t best_num_searched = UINT_MAX;         // NOTE: num_searched の最小値？
	TTEntry* best_entry_include_mate = nullptr;
	uint32_t best_num_searched_include_mate = UINT_MAX;    // NOTE
	                                                       //     : 詰み局面のみのnum_searched 最小値？
                                       	                   //       名前的には違うっぽいけど。。。
	for (auto& entry : entries.entries) {
		if (entry.pn != 0) {
			if (best_num_searched > entry.num_searched) {
				best_entry = &entry;
				best_num_searched = entry.num_searched;
			}
		}
		else {
			if (best_num_searched_include_mate > entry.num_searched) {
				best_entry_include_mate = &entry;
				best_num_searched_include_mate = entry.num_searched;
			}
		}
	}
	// NOTE
	//     : 詰みを証明できている局面の価値は高いので、
	//       詰みが証明できていない局面が一つも無い時にのみ、詰みが証明できている局面を潰す。
	if (best_entry == nullptr)
		best_entry = best_entry_include_mate;

	best_entry->hash_high = hash_high;
	best_entry->hand = hand;
	best_entry->depth = depth;
	best_entry->pn = 1;
	best_entry->dn = 1;
	best_entry->generation = generation;
	best_entry->num_searched = 0;
	return *best_entry;
}

template <bool or_node>
TTEntry& TranspositionTable::LookUp(const Position& n) {
	// NOTE: このLookUp<>() はLookUp() のラッパー
	return LookUp(n.getBoardKey(), or_node ? n.hand(n.turn()) : n.hand(oppositeColor(n.turn())), n.gamePly());
}

// moveを指した後の子ノードのキーを返す
template <bool or_node>
void TranspositionTable::GetChildFirstEntry(const Position& n, const Move move, Cluster*& entries, uint32_t& hash_high, Hand& hand) {
	// 手駒は常に先手の手駒で表す
	if (or_node) {
		hand = n.hand(n.turn());
		if (move.isDrop()) {
			hand.minusOne(move.handPieceDropped());
		}
		else {
			const Piece to_pc = n.piece(move.to());
			if (to_pc != Empty) {
				const PieceType pt = pieceToPieceType(to_pc);
				hand.plusOne(pieceTypeToHandPiece(pt));
			}
		}
	}
	else {
		hand = n.hand(oppositeColor(n.turn()));
	}
	Key key = n.getBoardKeyAfter(move);
	entries = &tt[key & clusters_mask];
	hash_high = key >> 32;
}

// moveを指した後の子ノードの置換表エントリを返す
template <bool or_node>
TTEntry& TranspositionTable::LookUpChildEntry(const Position& n, const Move move) {
	Cluster* entries;
	uint32_t hash_high;
	Hand hand;
	GetChildFirstEntry<or_node>(n, move, entries, hash_high, hand);
	return LookUpDirect(*entries, hash_high, hand, n.gamePly() + 1);
}

void TranspositionTable::Resize(int64_t hash_size_mb) {
	if (hash_size_mb == 16) {
		// TODO: どういうこと？
		hash_size_mb = 4096;
	}
	// NOTE
	//     : msb = Most Significant Bit = 最上位bit
	//       つまり、確保できる限界容量に限りなく近づけつつ、hash の縦の列のサイズを2のn乗 にしたい、って感じかな。
	//       恐らくhash のアクセス時のkey を計算する際のmask(=clusters_mask) がpopcnt == 1 である単純なbit に出来るからかな？
	int64_t new_num_clusters = 1LL << msb((hash_size_mb * 1024 * 1024) / sizeof(Cluster));
	if (new_num_clusters == num_clusters) {
		return;
	}

	num_clusters = new_num_clusters;

	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}

	// NOTE: OverFlow とか桁落ちとかを防ぐために、割って掛けて割ってる
	std::cout << "info string alloc_size=" << (sizeof(Cluster) * (new_num_clusters / 1024)) / 1024 << "mb" << std::endl;

	// NOTE
	//     : 特に型は指定せずに、ただただ(new_num_clusters * sizeof(Cluster) + CacheLineSize) * (1) byte の領域を確保
	// TODO
	//     : CacheLineSize とは？
	//     : ((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1)) って何してる？
	tt_raw = std::calloc(new_num_clusters * sizeof(Cluster) + CacheLineSize, 1);
	tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));
	clusters_mask = num_clusters - 1;
}

void TranspositionTable::NewSearch() {
	++generation;
	if (generation == 0) generation = 1;
}

static const constexpr int kInfinitePnDn = 100000000;

// 王手の指し手が近接王手か
FORCE_INLINE bool moveGivesNeighborCheck(const Position& pos, const Move& move)
{
	const Color them = oppositeColor(pos.turn());
	const Square ksq = pos.kingSquare(them);

	const Square to = move.to();

	// 敵玉の8近傍
	if (pos.attacksFrom<King>(ksq).isSet(to))
		return true;

	// 桂馬による王手
	if (move.pieceTypeTo() == Knight)
		return true;

	return false;
}

// 反証駒を計算(持っている持ち駒を最大数にする(後手の持ち駒を加える))
FORCE_INLINE u32 dp(const Hand& us, const Hand& them) {
	u32 dp = 0;
	u32 pawn = us.exists<HPawn>(); if (pawn > 0) dp += pawn + them.exists<HPawn>();
	u32 lance = us.exists<HLance>(); if (lance > 0) dp += lance + them.exists<HLance>();
	u32 knight = us.exists<HKnight>(); if (knight > 0) dp += knight + them.exists<HKnight>();
	u32 silver = us.exists<HSilver>(); if (silver > 0) dp += silver + them.exists<HSilver>();
	u32 gold = us.exists<HGold>(); if (gold > 0) dp += gold + them.exists<HGold>();
	u32 bishop = us.exists<HBishop>(); if (bishop > 0) dp += bishop + them.exists<HBishop>();
	u32 rook = us.exists<HRook>(); if (rook > 0) dp += rook + them.exists<HRook>();
	return dp;
}

// @arg thpn, thdn
//     : 一番初めのroot ではinf が渡されるっぽい。
//     : この局面に入ってくる直前において、
//       Position n の兄弟局面 の内、2番目に良い局面のpn, dn
//       ※2番目に良い局面とは、
//         OrNode なら2番目にpn が小さい局面、
//         AndNode なら2番目にdn が小さい局面 のことを指す。
template <bool or_node>
void DfPn::dfpn_inner(Position& n, const int thpn, const int thdn/*, bool inc_flag*/, const uint16_t maxDepth, int64_t& searchedNode) {
	auto& entry = transposition_table.LookUp<or_node>(n);

	if (or_node) {
		// NOTE: 深さ制限に達したら、不詰みと判定
		if (n.gamePly() + 1 > maxDepth) {
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			entry.num_searched = REPEAT;    // TODO: REPEAT って何？
			return;    // NOTE: serach_result = 中断(depth limit)
		}
	}

	// if (n is a terminal node) { handle n and return; }
	MovePicker<or_node> move_picker(n);
	if (move_picker.empty()) {    // NOTE: 手がこれ以上続かず詰ませられ無い or 詰まされたので投了するしかない
		// nが先端ノード

		if (or_node) {
			// 自分の手番でここに到達した場合は王手の手が無かった、
			entry.pn = kInfinitePnDn;
			entry.dn = 0;

			// 反証駒
			// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
			entry.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
		}
		else {
			// 相手の手番でここに到達した場合は王手回避の手が無かった、
			// 1手詰めを行っているため、ここに到達することはない
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
		}

		return;    // NOTE: serach_result = 不詰み || 詰み
	}
	//// NOTE
    ////     : 一旦、2 or 3手詰め 探索をする。
	// 新規節点で固定深さの探索を併用
	if (entry.num_searched == 0) {
		if (or_node) {
			// 3手詰みチェック
			Color us = n.turn();
			Color them = oppositeColor(us);

			StateInfo si;
			StateInfo si2;

			const CheckInfo ci(n);
			for (const auto& ml : move_picker)
			{
				const Move& m = ml.move;

				n.doMove(m, si, ci, true);

				// 千日手のチェック
				if (n.isDraw(16) == RepetitionWin) {
					// NOTE
					//     : 詰将棋用のMovePicker で千日手になったなら、連続王手の千日手だから...?
					// 受け側の反則勝ち
					n.undoMove(m);
					continue;
				}

				auto& entry2 = transposition_table.LookUp<false>(n);

				// この局面ですべてのevasionを試す
				MovePicker<false> move_picker2(n);

				if (move_picker2.size() == 0) {
					// NOTE
					//     : 後手がgameover 故、今回の指し手m は詰ませる手である。
					// 1手で詰んだ
					n.undoMove(m);

					// TODO
					//     : kInfinitePnDn + 1 と+1 する理由は何？
					entry2.pn = 0;
					entry2.dn = kInfinitePnDn + 1;

					entry.pn = 0;
					entry.dn = kInfinitePnDn;

					// NOTE
					//     : 定義通りにするなら、詰んだ局面であるentry2 での証明駒 = 0 で、
					//       m が駒を打つ手ならば "entry での証明駒 = entry.hand + 打った駒" と差分更新
					// 証明駒を初期化
					entry.hand.set(0);

					// 打つ手ならば証明駒に加える
					if (m.isDrop()) {
						entry.hand.plusOne(m.handPieceDropped());
					}
					// 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
					if (!moveGivesNeighborCheck(n, m))
						entry.hand.setPP(n.hand(n.turn()), n.hand(oppositeColor(n.turn())));

					// NOTE: root のOrNode で詰みを見つけたので終了
					return;    // NOTE: serach_result = 詰み
				}

				// NOTE: 手数限界を突破したので、不詰み扱い
				if (n.gamePly() + 2 > maxDepth) {
					n.undoMove(m);

					entry2.pn = kInfinitePnDn;
					entry2.dn = 0;
					entry2.num_searched = REPEAT;

					continue;
				}

				// NOTE
				//     : 今は遷移してAndNode に居るので、一つでも詰みを逃れる指し手があれば、NEXT_CHECK へとジャンプする。
				const CheckInfo ci2(n);
				for (const auto& move : move_picker2)
				{
					const Move& m2 = move.move;

					// この指し手で逆王手になるなら、不詰めとして扱う
					if (n.moveGivesCheck(m2, ci2))
						goto NEXT_CHECK;

					n.doMove(m2, si2, ci2, false);

					if (n.mateMoveIn1Ply<true>()) {
						auto& entry1 = transposition_table.LookUp<true>(n);
						entry1.pn = 0;
						entry1.dn = kInfinitePnDn + 2;
					}
					else {
						// 詰んでないので、m2で詰みを逃れている。
						n.undoMove(m2);
						goto NEXT_CHECK;
					}

					n.undoMove(m2);
				}

				// すべて詰んだ
				n.undoMove(m);

				entry2.pn = 0;
				entry2.dn = kInfinitePnDn;

				entry.pn = 0;
				entry.dn = kInfinitePnDn;

				return;

			NEXT_CHECK:;
				n.undoMove(m);

				if (entry2.num_searched == 0) {
					// TODO
					//     : 何故pn, dn が下記の値になる？
					//       pn はまぁ分かる気もするが、dn は何故?
					//       いや、あれか、AndNode だから、pn = move_picker2.size() なのは良いか。(ただこれ、子ノード展開しないといけないのでは？)
					//       だけど、dnは dn = 1; とすべきじゃないの？普通にするならさ？
					//     : そもそも、何故num_searched == 0 の時だけ？
					entry2.num_searched = 1;
					entry2.pn = static_cast<int>(move_picker2.size());
					entry2.dn = static_cast<int>(move_picker2.size());
				}
			}
		}
		else {
			// 2手読みチェック
			StateInfo si2;
			// この局面ですべてのevasionを試す
			const CheckInfo ci2(n);
			for (const auto& move : move_picker)
			{
				const Move& m2 = move.move;

				// この指し手で逆王手になるなら、不詰めとして扱う
				if (n.moveGivesCheck(m2, ci2))
					goto NO_MATE;

				n.doMove(m2, si2, ci2, false);

				// TODO
				//     : dlshogi にならって、
				//       template <bool Additional = true> Move mateMoveIn1Ply(); を
				//       template <bool Additional = false> Move mateMoveIn1Ply(); に。
				//       (正確には、dlshogi はAdditional が無い。)
				if (const Move move = n.mateMoveIn1Ply<true>()) {
					auto& entry1 = transposition_table.LookUp<true>(n);
					entry1.pn = 0;
					entry1.dn = kInfinitePnDn + 2;

					// 証明駒を初期化
					entry1.hand.set(0);

					// 打つ手ならば証明駒に加える
					if (move.isDrop()) {
						entry1.hand.plusOne(move.handPieceDropped());
					}
					// 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
					if (!moveGivesNeighborCheck(n, move))
						entry1.hand.setPP(n.hand(n.turn()), n.hand(oppositeColor(n.turn())));
				}
				else {
					// 詰んでないので、m2で詰みを逃れている。
					// 不詰みチェック
					// 王手がない場合
					MovePicker<true> move_picker2(n);
					if (move_picker2.empty()) {
						auto& entry1 = transposition_table.LookUp<true>(n);
						entry1.pn = kInfinitePnDn;
						entry1.dn = 0;
						// 反証駒
						// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
						entry1.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));

						n.undoMove(m2);

						entry.pn = kInfinitePnDn;
						entry.dn = 0;
						// 子局面の反証駒を設定
						// 打つ手ならば、反証駒から削除する
						if (m2.isDrop()) {
							entry.hand = entry1.hand;
							entry.hand.minusOne(m2.handPieceDropped());
						}
						// 先手の駒を取る手ならば、反証駒に追加する
						else {
							const Piece to_pc = n.piece(m2.to());
							if (to_pc != Empty) {
								const PieceType pt = pieceToPieceType(to_pc);
								const HandPiece hp = pieceTypeToHandPiece(pt);
								if (entry.hand.numOf(hp) > entry1.hand.numOf(hp)) {
									entry.hand = entry1.hand;
									entry.hand.plusOne(hp);
								}
							}
						}
						return;
					}
					n.undoMove(m2);
					goto NO_MATE;
				}

				n.undoMove(m2);
			}

			// すべて詰んだ
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			return;

		NO_MATE:;
		}
	}

	// TODO
	//     : どうしてこのタイミングで千日手判定するんや？
	//       -> 恐らくやけど、dfpn 呼ぶ局面では詰みの方が珍しいからやないかな。
	//          (先に短手数の詰みを確認しておいた方が、この関数を抜けると期待される時間は小さくなる。)
	// 千日手のチェック
	switch (n.isDraw(16)) {
	case RepetitionWin:
		//cout << "RepetitionWin" << endl;
		// 連続王手の千日手による勝ち
		if (or_node) {
			// ここは通らないはず
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			entry.num_searched = REPEAT;
		}
		else {
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			entry.num_searched = REPEAT;
		}
		return;

	case RepetitionLose:
		//cout << "RepetitionLose" << endl;
		// 連続王手の千日手による負け
		if (or_node) {
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			entry.num_searched = REPEAT;
		}
		else {
			// ここは通らないはず
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			entry.num_searched = REPEAT;
		}
		return;

	case RepetitionDraw:
		//cout << "RepetitionDraw" << endl;
		// 普通の千日手
		// ここは通らないはず
		entry.pn = kInfinitePnDn;
		entry.dn = 0;
		entry.num_searched = REPEAT;
		return;

	case RepetitionSuperior:
		if (!or_node) {
			// NOTE
			//     : AndNode で優越局面になった場合、AndNode 視点で有利になってる訳で、
			//       OrNode 視点ではこの局面に遷移すべきでない。
			//       従って、OrNode 視点で"マイナスの評価値" となるようにpn = inf とする。
			// ANDノードで優越局面になっている場合、除外できる(ORノードで選択されなくなる)
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			entry.num_searched = REPEAT;
			return;
		}
		break;
	}

	// NOTE
	//     : 以下の情報があれば、置換表に直接アクセス(LookUpDirect)出来る。
	// 子局面のハッシュエントリをキャッシュ
	struct TTKey {
		TranspositionTable::Cluster* entries;
		uint32_t hash_high;
		Hand hand;
	} ttkeys[MaxCheckMoves];

	for (const auto& move : move_picker) {
		auto& ttkey = ttkeys[&move - move_picker.begin()];
		transposition_table.GetChildFirstEntry<or_node>(n, move, ttkey.entries, ttkey.hash_high, ttkey.hand);
	}

	// NOTE
	//     : 任意のnode の探索数はmaxSearchNode を超えてはならない。
	while (searchedNode < maxSearchNode && !stop) {
		// NOTE: 恐らくnum_searched について代入でない更新がなされるのはここだけ。
		++entry.num_searched;

		Move best_move;
		// NOTE
		//     : 再帰する時に、thpn, thdn に渡す値で。
		//       子ノードをroot とする探索に影響。
		//     : これらは最後にset するので、一度set された後は実装上実質的にconst
		int thpn_child;
		int thdn_child;

		// NOTE
		//     : 現在見ている局面の情報(=entry) を、子ノードの情報(=child_entry) を元に更新
		//     : 探索経路が初めから終わりまで一本の長い線で一筆書き出来るので、
		//       PUCT みたいにわざわざbackup とかせずとも、子ノードのpn, dn を集計するだけで
		//       現在のnode のpn, dn を更新可能って感じ？
		//       -> いや、うーん、なんか理由が微妙に違う気がするな。
		// expand and compute pn(n) and dn(n);
		if (or_node) {
			// ORノードでは、最も証明数が小さい = 玉の逃げ方の個数が少ない = 詰ましやすいノードを選ぶ
			int best_pn = kInfinitePnDn;
			int second_best_pn = kInfinitePnDn;    // NOTE: thpn_child に使うらしい。。。
			int best_dn = 0;    // NOTE: best_pn が最小を記録するたびに更新するので、
			                    //       pn, dn の間には凡そ反比例に関係が成り立つと仮定している模様。
			                    //       つまり、best_dn には、dn の最大値の近似値が格納されるはず。
			uint32_t best_num_search = UINT_MAX;    // NOTE: best_pn な子ノードのnum_searched 

			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			// 子局面の反証駒の積集合
			u32 pawn = UINT_MAX;
			u32 lance = UINT_MAX;
			u32 knight = UINT_MAX;
			u32 silver = UINT_MAX;
			u32 gold = UINT_MAX;
			u32 bishop = UINT_MAX;
			u32 rook = UINT_MAX;
			bool repeat = false; // 最大手数チェック用    // TODO: これなんやねん。
			for (const auto& move : move_picker) {
				auto& ttkey = ttkeys[&move - move_picker.begin()];
				const auto& child_entry = transposition_table.LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				// NOTE: OrNode で子ノードの一つが詰みだったので、現在のノードも詰み
				if (child_entry.pn == 0) {
					// 詰みの場合
					//cout << n.toSFEN() << " or" << endl;
					//cout << bitset<32>(entry.hand.value()) << endl;
					entry.pn = 0;
					entry.dn = kInfinitePnDn;
					// 子局面の証明駒を設定
					// 打つ手ならば、証明駒に追加する
					if (move.move.isDrop()) {
						const HandPiece hp = move.move.handPieceDropped();    // NOTE: move で打った手
						// NOTE
						//     : 証明駒は小さい方が良いので、打った駒hp が証明駒に何枚必要かを最小化する。
						//       entry.hand (現状のhp の証明駒中の数の最小) vs entry.hand.plusOne(hp) (今回で判明した hpの証明駒中の数) で比較して、より小さい方を採用。
						//       この時、hp だけでなく、持ち駒全体を更新している事に注意。
						//       (hp の数だけを変更すると、今までが指し手Aを想定した証明駒で、今回のが指し手B(=move)を想定した証明駒の時に違法である。)
						// TODO: これ、hp だけで判断せずに、証明駒の総数とかで、どっち採用するか決めた方が良いよね？
						if (entry.hand.numOf(hp) > child_entry.hand.numOf(hp)) {
							entry.hand = child_entry.hand;
							entry.hand.plusOne(move.move.handPieceDropped());    // TODO: 何故、これ、.plusOne(hp); にしない？
						}
					}
					// TODO
					//     : 駒を取る場合は、問答無用で今回の指し手についての証明駒を採用するっぽい。何故？？
					// 後手の駒を取る手ならば、証明駒から削除する
					else {
						const Piece to_pc = n.piece(move.move.to());
						if (to_pc != Empty) {    // NOTE: 駒を取る手である
							entry.hand = child_entry.hand;
							const PieceType pt = pieceToPieceType(to_pc);
							const HandPiece hp = pieceTypeToHandPiece(pt);
							if (entry.hand.exists(hp))
								entry.hand.minusOne(hp);
						}
					}
					//cout << bitset<32>(entry.hand.value()) << endl;
					// NOTE
					//     : 今はOrNode 、つまり詰みについてOr であり、詰みが見つかったのでbreak
					break;
				}
				// NOTE
				//     : この関数のroot node(entry に対応した局面) が不詰みなら、毎度ここを通る(はず)。
				//       (一応、置換表から貰った別の局面では不詰みと言い切れたけど、この局面では違うパターンもあって、その場合は上のif に入る場合がある。)
				//     : 以下のように、反証駒が登録されてる任意の子ノードの、任意の反証駒(ex 歩, 香, 桂馬, ...) についての最小値を求めれば、それが則ち積集合である。
				else if (entry.dn == 0) {
					if (child_entry.dn == 0) {
						const Hand& child_dp = child_entry.hand;
						// 歩
						const u32 child_pawn = child_dp.exists<HPawn>();
						if (child_pawn < pawn) pawn = child_pawn;
						// 香車
						const u32 child_lance = child_dp.exists<HLance>();
						if (child_lance < lance) lance = child_lance;
						// 桂馬
						const u32 child_knight = child_dp.exists<HKnight>();
						if (child_knight < knight) knight = child_knight;
						// 銀
						const u32 child_silver = child_dp.exists<HSilver>();
						if (child_silver < silver) silver = child_silver;
						// 金
						const u32 child_gold = child_dp.exists<HGold>();
						if (child_gold < gold) gold = child_gold;
						// 角
						const u32 child_bishop = child_dp.exists<HBishop>();
						if (child_bishop < bishop) bishop = child_bishop;
						// 飛車
						const u32 child_rook = child_dp.exists<HRook>();
						if (child_rook < rook) rook = child_rook;
					}
				}
				// NOTE: 進研ゼミでやったところ
				entry.pn = std::min(entry.pn, child_entry.pn);
				entry.dn += child_entry.dn;

				// 最大手数で不詰みの局面が優越関係で使用されないようにする
				if (child_entry.dn == 0 && child_entry.num_searched == REPEAT)
					repeat = true;

				// NOTE
				//     : 証明数がより小さい or
				//       証明数が同じなら探索数がより少ない
				// TODO
				//     : 何故、child_entry.num_searched の探索数が少ない方が良いの？
				//       これまでの探索数が少ない方が、同じ証明数でもより詰みやすそうって話？
				//       -> 仮にそうだとしたら、逆にこれまでの探索では詰みが証明しやすい局面を探索し切ったので、
				//          これからの探索では難しい局面が残ってる説ってない？dfpn の性質的にも楽そうな奴から探索する訳やしさ？
				if (child_entry.pn < best_pn ||
					child_entry.pn == best_pn && best_num_search > child_entry.num_searched) {
					second_best_pn = best_pn;
					best_pn = child_entry.pn;
					best_dn = child_entry.dn;
					best_move = move;
					best_num_search = child_entry.num_searched;
				}
				else if (child_entry.pn < second_best_pn) {
					second_best_pn = child_entry.pn;
				}
			}    // NOTE: for 終了

			// TODO
			//     : kInfinitePnDn がどういう場面で代入されるか 把握できてない。
			//     : これは、子ノードに複数dn = kInfinitePnDn な奴が合った場合にclip するのが狙い？
			entry.dn = std::min(entry.dn, kInfinitePnDn);

			// NOTE
			//     : OrNode は不詰みに関してAnd なので、全部見てからでないとdn == 0 は途中では言い切れない。
			//       全部見終わったので、ここで初めて"dn == 0 なら不詰みである" が真となる。
			//     : df-pn の鉄則「詰み || 不詰み が判明したら、証明駒 || 反証駒 を計算して置換表にset」
			if (entry.dn == 0) {
				// 不詰みの場合
				//cout << n.hand(n.turn()).value() << "," << entry.hand.value() << ",";
				// NOTE
				//     : bool repeat = false; なるフラグについてのコメントや、以下のコメントなどを見るに、
				//       REPEAT とは最大手数で探索中断したことを示すフラグなのかもしれんな。
				// 最大手数で不詰みの局面が優越関係で使用されないようにする
				if (repeat)
					entry.num_searched = REPEAT;
				else {
					// 先手が一枚も持っていない種類の先手の持ち駒を反証駒から削除する
					u32 curr_pawn = entry.hand.template exists<HPawn>(); if (curr_pawn == 0) pawn = 0; else if (pawn < curr_pawn) pawn = curr_pawn;
					u32 curr_lance = entry.hand.template exists<HLance>(); if (curr_lance == 0) lance = 0; else if (lance < curr_lance) lance = curr_lance;
					u32 curr_knight = entry.hand.template exists<HKnight>(); if (curr_knight == 0) knight = 0; else if (knight < curr_knight) knight = curr_knight;
					u32 curr_silver = entry.hand.template exists<HSilver>(); if (curr_silver == 0) silver = 0; else if (silver < curr_silver) silver = curr_silver;
					u32 curr_gold = entry.hand.template exists<HGold>(); if (curr_gold == 0) gold = 0; else if (gold < curr_gold) gold = curr_gold;
					u32 curr_bishop = entry.hand.template exists<HBishop>(); if (curr_bishop == 0) bishop = 0; else if (bishop < curr_bishop) bishop = curr_bishop;
					u32 curr_rook = entry.hand.template exists<HRook>(); if (curr_rook == 0) rook = 0; else if (rook < curr_rook) rook = curr_rook;
					// 反証駒に子局面の証明駒の積集合を設定
					entry.hand.set(pawn | lance | knight | silver | gold | bishop | rook);
					//cout << entry.hand.value() << endl;
				}
			}
			else {
				// NOTE
				//     : if (entry.pn >= thpn || entry.dn >= thdn) { break; } と評価しており、
				//       第二候補の兄弟局面のスコアを超過して初めて打ち切りたいので、"+1" する。
				thpn_child = std::min(thpn, second_best_pn + 1);
				thdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn);
			}
		}
		else {
			// ANDノードでは最も反証数の小さい = 王手の掛け方の少ない = 不詰みを示しやすいノードを選ぶ
			int best_dn = kInfinitePnDn;
			int second_best_dn = kInfinitePnDn;
			int best_pn = 0;
			uint32_t best_num_search = UINT_MAX;

			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			// 子局面の証明駒の和集合
			u32 pawn = 0;
			u32 lance = 0;
			u32 knight = 0;
			u32 silver = 0;
			u32 gold = 0;
			u32 bishop = 0;
			u32 rook = 0;
			bool all_mate = true;    // NOTE: これまでに処理した子ノードの全てが詰みである限りtrue
			for (const auto& move : move_picker) {
				auto& ttkey = ttkeys[&move - move_picker.begin()];
				const auto& child_entry = transposition_table.LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				if (all_mate) {
					if (child_entry.pn == 0) {
						// NOTE: 証明駒の和集合を求める。
						const Hand& child_pp = child_entry.hand;
						// 歩
						const u32 child_pawn = child_pp.exists<HPawn>();
						if (child_pawn > pawn) pawn = child_pawn;
						// 香車
						const u32 child_lance = child_pp.exists<HLance>();
						if (child_lance > lance) lance = child_lance;
						// 桂馬
						const u32 child_knight = child_pp.exists<HKnight>();
						if (child_knight > knight) knight = child_knight;
						// 銀
						const u32 child_silver = child_pp.exists<HSilver>();
						if (child_silver > silver) silver = child_silver;
						// 金
						const u32 child_gold = child_pp.exists<HGold>();
						if (child_gold > gold) gold = child_gold;
						// 角
						const u32 child_bishop = child_pp.exists<HBishop>();
						if (child_bishop > bishop) bishop = child_bishop;
						// 飛車
						const u32 child_rook = child_pp.exists<HRook>();
						if (child_rook > rook) rook = child_rook;
					}
					// NOTE: 不詰みが見つかったので、all_mate はfalse となる。
					else {
						all_mate = false;
					}
				}
				if (child_entry.dn == 0) {
					// 不詰みの場合
					entry.pn = kInfinitePnDn;
					entry.dn = 0;
					// 最大手数で不詰みの局面が優越関係で使用されないようにする
					if (child_entry.num_searched == REPEAT)
						entry.num_searched = REPEAT;
					else {
						// 子局面の反証駒を設定
						// 打つ手ならば、反証駒から削除する
						if (move.move.isDrop()) {
							const HandPiece hp = move.move.handPieceDropped();
							if (entry.hand.numOf(hp) < child_entry.hand.numOf(hp)) {
								entry.hand = child_entry.hand;
								entry.hand.minusOne(hp);
							}
						}
						// 先手の駒を取る手ならば、反証駒に追加する
						else {
							const Piece to_pc = n.piece(move.move.to());
							if (to_pc != Empty) {
								const PieceType pt = pieceToPieceType(to_pc);
								const HandPiece hp = pieceTypeToHandPiece(pt);
								if (entry.hand.numOf(hp) > child_entry.hand.numOf(hp)) {
									entry.hand = child_entry.hand;
									entry.hand.plusOne(hp);
								}
							}
						}
					}
					break;
				}
				entry.pn += child_entry.pn;
				entry.dn = std::min(entry.dn, child_entry.dn);

				if (child_entry.dn < best_dn ||
					child_entry.dn == best_dn && best_num_search > child_entry.num_searched) {
					second_best_dn = best_dn;
					best_dn = child_entry.dn;
					best_pn = child_entry.pn;
					best_move = move;
				}
				else if (child_entry.dn < second_best_dn) {
					second_best_dn = child_entry.dn;
				}
			}    // NOTE: for 終了

			entry.pn = std::min(entry.pn, kInfinitePnDn);

			// NOTE
			//     : ここで初めて"pn == 0 ならば詰みである" が真となる。
			//     : df-pn の鉄則より、反証駒の計算とset
			if (entry.pn == 0) {
				// 詰みの場合
				//cout << n.toSFEN() << " and" << endl;
				//cout << bitset<32>(entry.hand.value()) << endl;
				// 証明駒に子局面の証明駒の和集合を設定
				u32 curr_pawn = entry.hand.template exists<HPawn>(); if (pawn > curr_pawn) pawn = curr_pawn;
				u32 curr_lance = entry.hand.template exists<HLance>(); if (lance > curr_lance) lance = curr_lance;
				u32 curr_knight = entry.hand.template exists<HKnight>(); if (knight > curr_knight) knight = curr_knight;
				u32 curr_silver = entry.hand.template exists<HSilver>(); if (silver > curr_silver) silver = curr_silver;
				u32 curr_gold = entry.hand.template exists<HGold>(); if (gold > curr_gold) gold = curr_gold;
				u32 curr_bishop = entry.hand.template exists<HBishop>(); if (bishop > curr_bishop) bishop = curr_bishop;
				u32 curr_rook = entry.hand.template exists<HRook>(); if (rook > curr_rook) rook = curr_rook;
				entry.hand.set(pawn | lance | knight | silver | gold | bishop | rook);
				//cout << bitset<32>(entry.hand.value()) << endl;

				// NOTE: 進研ゼミでやったところ
				// 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
				if (!(n.checkersBB() & n.attacksFrom<King>(n.kingSquare(n.turn())) || n.checkersBB() & n.attacksFrom<Knight>(n.turn(), n.kingSquare(n.turn()))))
					entry.hand.setPP(n.hand(oppositeColor(n.turn())), n.hand(n.turn()));
				//cout << bitset<32>(entry.hand.value()) << endl;
			}
			else {
				// TODO
				//     : このように更新する意味は何？？？？
				//       -> 前回のOrNode における最善&次善の兄弟局面についてのpnの差をΔPnPrev_12、
				//          次回のOrNode における最善&次善の兄弟局面についてのpnの差をΔPnNext_12 とし、
				//          次回のOrNode における最善の局面(子ノード, 指し手) のpn をPn_1 とすると、
				//          次回のOrNode において、thpn_child = Pn_1 + min(ΔPnPrev_12, ΔPnNext_12) + 1 とすることに等しい。
				//          特に、min(ΔPnPrev_12, ΔPnNext_12) == ΔPnNext_12 の場合、thpn_child = second_best_pn + 1 とすることに等しい。
				thpn_child = std::min(thpn - entry.pn + best_pn, kInfinitePnDn);
				thdn_child = std::min(thdn, second_best_dn + 1);
			}
		}

		// if (pn(n) >= thpn || dn(n) >= thdn)
		//   break; // termination condition is satisfied
		if (entry.pn >= thpn || entry.dn >= thdn) {
			break;
		}

		StateInfo state_info;
		//cout << n.toSFEN() << "," << best_move.toUSI() << endl;
		n.doMove(best_move, state_info);
		++searchedNode;
		dfpn_inner<!or_node>(n, thpn_child, thdn_child/*, inc_flag*/, maxDepth, searchedNode);
		n.undoMove(best_move);
	}
}

// 詰みの手返す
Move DfPn::dfpn_move(Position& pos) {
	MovePicker<true> move_picker(pos);
	for (const auto& move : move_picker) {
		const auto& child_entry = transposition_table.LookUpChildEntry<true>(pos, move);
		if (child_entry.pn == 0) {
			return move;
		}
	}

	return Move::moveNone();
}

template<bool or_node>
int DfPn::get_pv_inner(Position& pos, std::vector<Move>& pv) {
	if (or_node) {
		// ORノードで詰みが見つかったらその手を選ぶ
		MovePicker<true> move_picker(pos);
		for (const auto& move : move_picker) {
			const auto& child_entry = transposition_table.LookUpChildEntry<true>(pos, move);
			if (child_entry.pn == 0) {
				if (child_entry.dn == kInfinitePnDn + 1) {
					pv.emplace_back(move);
					return 1;
				}
				StateInfo state_info;
				pos.doMove(move, state_info);
				switch (pos.isDraw(16)) {
				// NOTE: 以下のどちらかであれば再帰。
				case NotRepetition:
				case RepetitionSuperior:
				{
					pv.emplace_back(move);
					const auto depth = get_pv_inner<false>(pos, pv);
					pos.undoMove(move);
					return depth + 1;
				}
				default:
					break;
				}
				pos.undoMove(move);
			}
		}
	}
	else {
		// ANDノードでは詰みまでが最大手数となる手を選ぶ
		int max_depth = 0;
		std::vector<Move> max_pv;
		MovePicker<false> move_picker(pos);
		for (const auto& move : move_picker) {
			const auto& child_entry = transposition_table.LookUpChildEntry<false>(pos, move);
			if (child_entry.pn == 0) {
				std::vector<Move> tmp_pv{ move };
				StateInfo state_info;
				pos.doMove(move, state_info);
				int depth = -kInfinitePnDn;
				if (child_entry.dn == kInfinitePnDn + 2) {
					depth = 1;
					if (!pos.inCheck()) {
						// 1手詰みチェック
						Move mate1ply = pos.mateMoveIn1Ply<true>();
						if (mate1ply) {
							tmp_pv.emplace_back(mate1ply);
						}
					}
					else
						get_pv_inner<true>(pos, tmp_pv);
				}
				else {
					depth = get_pv_inner<true>(pos, tmp_pv);
				}
				pos.undoMove(move);

				if (depth > max_depth) {
					max_depth = depth;
					max_pv = std::move(tmp_pv);
				}
			}
		}
		if (max_depth > 0) {
			std::copy(max_pv.begin(), max_pv.end(), std::back_inserter(pv));
			return max_depth + 1;
		}
	}
	return -kInfinitePnDn;
}

// PVと詰みの手返す
std::tuple<std::string, int, Move> DfPn::get_pv(Position& pos) {
	std::vector<Move> pv;
	const auto depth = get_pv_inner<true>(pos, pv);
	const Move& best_move = pv[0];
	std::stringstream ss;
	ss << best_move.toUSI();
	for (size_t i = 1; i < pv.size(); i++)
		ss << " " << pv[i].toUSI();

	return std::make_tuple(ss.str(), depth, best_move);
}

void DfPn::init()
{
	transposition_table.Resize(HASH_SIZE_MB);
}

// 詰将棋探索のエントリポイント
bool DfPn::dfpn(Position& r) {
	// キャッシュの世代を進める

	//std::cout << "debug: DLSHOGI dfpn::dfpn" << std::endl;

	transposition_table.NewSearch();

	searchedNode = 0;    // NOTE: 探索ノード数をreset
	if (!r.inCheck()) {
		// 1手詰みチェック
		Move mate1ply = r.mateMoveIn1Ply<true>();
		if (mate1ply) {
			auto& child_entry = transposition_table.LookUpChildEntry<true>(r, mate1ply);
			child_entry.pn = 0;
			child_entry.dn = kInfinitePnDn + 1;
			return true;
		}
	}
	dfpn_inner<true>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, std::min(r.gamePly() + kMaxDepth, draw_ply), searchedNode);
	const auto& entry = transposition_table.LookUp<true>(r);

	//cout << searchedNode << endl;

	/*std::vector<Move> moves;
	std::unordered_set<Key> visited;
	dfs(true, r, moves, visited);
	for (Move& move : moves)
	cout << move.toUSI() << " ";
	cout << endl;*/

	return entry.pn == 0;
}

// 詰将棋探索のエントリポイント
bool DfPn::dfpn_andnode(Position& r) {
	// 自玉に王手がかかっていること

	// キャッシュの世代を進める
	transposition_table.NewSearch();

	searchedNode = 0;
	dfpn_inner<false>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, std::min(r.gamePly() + kMaxDepth, draw_ply), searchedNode);
	const auto& entry = transposition_table.LookUp<false>(r);

	return entry.pn == 0;
}

#endif    // DFPN_DLSHOGI