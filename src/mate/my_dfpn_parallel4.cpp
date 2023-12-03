#include <unordered_set>

#include "my_dfpn_parallel4.hpp"

#include "debug_string_queue.hpp"


#ifdef DFPN_PARALLEL4

//constexpr int skipSize[]  = { 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
//constexpr int skipPhase[] = { 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

constexpr int skipSize[]  = { 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
constexpr int skipPhase[] = { 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

//// debug
// 以下のハッシュの局面のpn が0 又はその可能性がある変数 に上書きされた場合にqueue_debug_str() する。
//constexpr uint64_t DEBUG_HASH_0 = 14886559628031884718ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_0 = DEBUG_HASH_0 >> 32;
//constexpr uint64_t DEBUG_HASH_0 = 3186861888099239777ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_0 = DEBUG_HASH_0 >> 32;
//constexpr uint64_t DEBUG_HASH_0 = 6962477625095860564ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_0 = DEBUG_HASH_0 >> 32;
//constexpr uint64_t DEBUG_HASH_0 = 709581966594841918ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_0 = DEBUG_HASH_0 >> 32;
//constexpr uint64_t DEBUG_HASH_0 = 2905829266871174452ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_0 = DEBUG_HASH_0 >> 32;
//
//constexpr uint64_t DEBUG_HASH_0_CH = 272799917983367539ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_0_CH = DEBUG_HASH_0_CH >> 32;

constexpr uint64_t DEBUG_HASH_0 = 12033080882891618798ULL;
constexpr uint64_t DEBUG_HASH_HIGH_0 = DEBUG_HASH_0 >> 32;
constexpr uint64_t DEBUG_HASH_P_0 = 2027170531019441307ULL;
constexpr uint64_t DEBUG_HASH_P_HIGH_0 = DEBUG_HASH_P_0 >> 32;

constexpr uint64_t DEBUG_HASH_1 = 15344768792386519950ULL;
constexpr uint64_t DEBUG_HASH_HIGH_1 = DEBUG_HASH_1 >> 32;
constexpr uint64_t DEBUG_HASH_P_1 = 9579420516929808861ULL;
constexpr uint64_t DEBUG_HASH_P_HIGH_1 = DEBUG_HASH_P_1 >> 32;

// skip 目的only
constexpr uint64_t DEBUG_HASH_2_SKIP = 9474161064609960782ULL;
constexpr uint64_t DEBUG_HASH_HIGH_2_SKIP = DEBUG_HASH_2_SKIP >> 32;
constexpr uint64_t DEBUG_HASH_P_2_SKIP = 7098768730445593323ULL;
constexpr uint64_t DEBUG_HASH_P_HIGH_2_SKIP = DEBUG_HASH_P_2_SKIP >> 32;


constexpr uint64_t DEBUG_HASH_3 = 17035644172859462960ULL;
constexpr uint64_t DEBUG_HASH_HIGH_3 = DEBUG_HASH_3 >> 32;
constexpr uint64_t DEBUG_HASH_P_3 = 3917466325995148055ULL;
constexpr uint64_t DEBUG_HASH_P_HIGH_3 = DEBUG_HASH_P_3 >> 32;

constexpr uint64_t DEBUG_HASH_4 = 13795971860294232562ULL;
constexpr uint64_t DEBUG_HASH_HIGH_4 = DEBUG_HASH_4 >> 32;
constexpr uint64_t DEBUG_HASH_P_4 = 901289320336155321ULL;
constexpr uint64_t DEBUG_HASH_P_HIGH_4 = DEBUG_HASH_P_4 >> 32;

constexpr uint64_t DEBUG_HASH_P_5 = 12456332060767114348ULL;
constexpr uint64_t DEBUG_HASH_P_HIGH_5 = DEBUG_HASH_P_5 >> 32;


constexpr uint64_t DEBUG_HASH_6 = 12456332060767114348ULL;
constexpr uint64_t DEBUG_HASH_HIGH_6 = DEBUG_HASH_6 >> 32;
constexpr uint64_t DEBUG_HASH_P_6 = 7127765833921877639ULL;
constexpr uint64_t DEBUG_HASH_P_HIGH_6 = DEBUG_HASH_P_6 >> 32;

constexpr uint64_t DEBUG_HASH_HIGH_TAG = 2900215811ULL;

// debug 時に出力する条件
//#define THREADID_COND (threadid == 1)


//// Impl
using namespace std;
using namespace ns_dfpn;

// MateMoveIn1Ply<>() のadditional
constexpr bool mmin1ply_Additional = true;

// 1 + epx
// https://link.springer.com/chapter/10.1007/978-3-319-09165-5_12
constexpr float EPS = 0;
constexpr float _EPS_PLUS_ONE = 1 + EPS;

// 1 も 3 も 4 も速度が落ちた
constexpr float THPN_MULTIPLY = 2;
constexpr float THDN_MULTIPLY = 2;

// TODO
//     : 同じmutex を獲得することが、同じCluster にアクセスすることの必要or必要十分 条件で無ければならない。
//       -> 則ち、同じmutex を獲得することが、同じCluster にアクセスすることの十分条件になってはならない。
//       -> 則ち、任意のCluster に対して、そのCluster にアクセス出来るmutex が一意に定まらなければならない。
//       -> 則ち、positinoMutex 用のbit mask はboardKeyの下位P bit、
//          Cluster についてのhask key 用のbit mask はboardKeyの下位C bit とすると、C >= P を満たさなければならない(はず)。
constexpr uint64_t POS_MUTEX_NUM = 65536; // must be 2^n
std::mutex g_pos_mutexes[POS_MUTEX_NUM];

// NOTE: thread safe (https://yohhoy.hatenablog.jp/entry/2013/12/15/204116)
// mutex
inline std::mutex& get_position_mutex(const Position* pos)
{
	return g_pos_mutexes[pos->getKey() & (POS_MUTEX_NUM - 1)];
}



// NOTE
//     : REPEAT って、千日手 or 最大手数で中断 の場合にset されて、そいつらを他と区別する為？
//int64_t ParallelDfPn::HASH_SIZE_MB = 2048;
//int ParallelDfPn::draw_ply = INT_MAX;
const constexpr uint32_t REPEAT = UINT_MAX - 1;

// --- 詰み将棋探索

void ParallelDfPn::dfpn_stop(const bool stop)
{
	this->stop = stop;
}

// NOTE: まじで以下のコメントの通りで、詰将棋する上であり得る指し手を全て列挙する。
// 詰将棋エンジン用のMovePicker
namespace ns_dfpn {
	// このentryをlockする。
	void TTEntry::lock() {
#ifdef DEBUG_LOCK_SAFE
		stopwatch_t sw;
		bool over_time_lim = false;
		sw.start();
#endif
		// NOTE: fales の時にtrue へとアトミックに書き換えが出来たら、その瞬間に排他的なロックが獲得できてる。
		// 典型的なCAS lock
		while (true)
		{
#ifdef DEBUG_LOCK_SAFE
			if (!over_time_lim) {
				const auto&& time = sw.elapsed_ms_interim();
				if (4000 < time) {
					std::cout << "Error: lock() is too long to wait." << std::endl;
					std::cout << "ErrorInfo: time = [" << time << "]" << std::endl;
					std::cout << "ErrorInfo: hash_high = [" << hash_high << "]" << std::endl;
					std::cout << "ErrorInfo: hand = [" << hand.value() << "]" << std::endl;
					std::cout << "ErrorInfo: pn = [" << pn << "]" << std::endl;
					std::cout << "ErrorInfo: dn = [" << dn << "]" << std::endl;
					std::cout << "ErrorInfo: depth = [" << depth << "]" << std::endl;
					std::cout << "ErrorInfo: generation = [" << generation << "]" << std::endl;
					std::cout << "ErrorInfo: num_searched = [" << num_searched << "]" << std::endl;
					over_time_lim = true;
				}
			}
#endif
			bool expected = false;
			if (_mtx.compare_exchange_weak(expected, true)) {
				//if (_mtx.compare_exchange_strong(expected, true)) {
#ifdef DEBUG_LOCK_SAFE
				sw.stop();
#endif
				break;
			}
		}
	}

	// このentryをunlockする。
	// lock済みであること。
	void TTEntry::unlock() {
#ifdef DEBUG_UNLOCK_SAFE
		if (!_mtx) {
			std::cout << "Error: when call unlock(), mtx must be true." << std::endl;
			std::cout << "ErrorInfo: hash_high = [" << hash_high << "]" << std::endl;
			std::cout << "ErrorInfo: hand = [" << hand.value() << "]" << std::endl;
			std::cout << "ErrorInfo: pn = [" << pn << "]" << std::endl;
			std::cout << "ErrorInfo: dn = [" << dn << "]" << std::endl;
			std::cout << "ErrorInfo: depth = [" << depth << "]" << std::endl;
			std::cout << "ErrorInfo: generation = [" << generation << "]" << std::endl;
			std::cout << "ErrorInfo: num_searched = [" << num_searched << "]" << std::endl;
			throw std::runtime_error("Error: inappropriate unlock.");
		}
#endif
		_mtx = false;
	}


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

// TODO
//     : このkey でposition mutex 獲得すればlock 出来るはず。
TTEntry& TranspositionTable::LookUp(const Key key, const Hand hand, const uint16_t depth) {
	auto& entries = tt[key & clusters_mask];    // TODO: これデータ競合あるよね。
	uint32_t hash_high = key >> 32;

#ifdef DEBUG
	if (hash_high == HASH_HIGH_DEBGU_TARGET3) {
		std::cout << "[" << key << "]" << std::endl;
	}
#endif

	return LookUpDirect(entries, hash_high, hand, depth);
}

// TODO
//     : 一応、entries にアクセスする為のkey でposition_mutex 取得すればスレッドセーフだが、
//       LookUpDirect 自体は、entries に置換表からアクセスする為のkey を知らないのでこの関数ではlock出来ない。
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

		entry.lock();

		// TODO
		//     : break 又はreturn で抜ける直前では必ずunlock()
		//       それ以外は使わなくなったらunlock();
		//     : どっかunlock() されてなかったら、数百万もテストすればどっかはデッドロック起こしてくれるでしょ。
		//       その時に、CAS lock の所で待ち時間長い場合に警告出せばよい。
		// NOTE
		//     : 今の世代に一致しないものは、ゴミと見なす？(利用せずに上書きする？)
		if (generation != entry.generation) {
			// NOTE: ここは全く問題無いはず。。。
			// 空のエントリが見つかった場合
			entry.hash_high = hash_high;
			entry.depth = depth;
			entry.hand = hand;
			// TODO
			//     : 最悪の場合を想定してmax を登録しておく？
			//       だとしたら途中までじゃなくてなんで全体の最大値にしないの？この辺りは雑で良いってこと？
			entry.pn = max_pn;
			entry.dn = max_dn;
			entry.generation = generation;
			entry.num_searched = 0;

			entry.unlock();
			return entry;

		}

		// TODO
		//     : hash と世代が一致するのは絶対に必要？ generation が進むタイミングは？(see dlshogi)
		if (hash_high == entry.hash_high && generation == entry.generation) {    // NOTE: 同じ局面と見なす必要条件
			if (hand == entry.hand && depth == entry.depth) {
				entry.unlock();
				// keyが合致するエントリを見つけた場合
				// 残りのエントリに優越関係を満たす局面があり証明済みの場合、それを返す
				int debug_i_tmp = i;
				for (i++; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
					TTEntry& entry_rest = entries.entries[i];
					entry_rest.lock();
					if (generation != entry_rest.generation) {
						// TODO
						//     : これcontinue じゃダメなんですか？
						//       -> 前から潰していくので、異なるgeneratino が現れたら、それは今のgeneration が一つ前で終了したということ。
						//          それゆえこれ以上走査しても無駄。
						entry_rest.unlock();
						break;
					}
					// TODO
					//     : ここから返る場合、もしdepth が異なるなら書き換えるべきでない。。。？
					if (hash_high == entry_rest.hash_high) {
						// NOTE
						//     : 詰ます側が、残りのentry の持ち駒以上に持ち駒を持っているなら上位互換で、詰むはず、的な話？
						//       -> key でアクセスしてるので、盤面だけで見ても一致するとは限らんよな？そのあたりどうなってん？
						//     : (詰むし、且つ、持ち駒も上位互換なのでこれを返す、とも見れる。つまり、このentry_rest の局面(の持ち駒)の方が汎用性が高い。)
						if (entry_rest.pn == 0) {    // NOTE:  詰みが証明済み
							if (hand.isEqualOrSuperior(entry_rest.hand) && entry_rest.num_searched != REPEAT) {

								// debug
								if (entry_rest.hash_high == 2900215811ULL) {
									std::stringstream ss;
									ss << "[LRest:" << entry_rest.pn << "," << entry_rest.dn
										<< "," << entry_rest.depth << "," << entry_rest.hand.value() << "," << entry_rest.generation << "," << i << "]";
									enqueue_debug_str(ss.str());
								}
								entry_rest.unlock();
								// NOTE
								//     : ここから返すentry は、更新はむやみやたらにしてはいけないはず。
								//       ここからは、現局面の持ち駒より汎用性の高い持ち駒の局面(一番汎用性が高い)の情報を参照してる。
								//       更新は必ず、同等の汎用性or現状超過の汎用性を持つ値に上書きされるはずなので、
								//       現局面の汎用性と同等の汎用性で上書きしてしまい、一番汎用性が高かった情報の汎用性を下げてしまう可能性がある。
								return entry_rest;
							}
						}
						else if (entry_rest.dn == 0) {    // NOTE: 不詰み証明済み
							if (entry_rest.hand.isEqualOrSuperior(hand) && entry_rest.num_searched != REPEAT) {
								entry_rest.unlock();
								return entry_rest;
							}
						}
					}
					entry_rest.unlock();
				}
				// debug
				entry.lock();
				if (entry.hash_high == 2900215811ULL) {
					std::stringstream ss;
					ss << "[LNotRest:" << entry.pn << "," << entry.dn
						<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "," << debug_i_tmp << "]";
					enqueue_debug_str(ss.str());
				}
				entry.unlock();
				// NOTE
				//     : より強いことが言えるentry が無ければ、先に見つかった奴を返す
				//       query として与えられた局面(の持ち駒)より汎用性の高いentry は見つからなかった。
				//       -> この場合、このentry なら、更新する時に、更新してはならない条件は存在しないはず。。。
				//          (∵更新は必ず、同等の汎用性or現状超過の汎用性を持つ値に上書きされるはず。
				//             なので、現状一番汎用性の高いentry の汎用性を下げる方向に更新することはない。)
				//          
				return entry;
			}
			// TODO
			//     : 何故hash さえ一致して、持ち駒の優越関係を満たす証明済み局面があればそれを返す、にせずに、
			//       generation が一致することを確認してるの？
			//       -> L107 のif (generation != entry.generation) { からの節を見ても分かるように、
			//          前の古い世代の探索結果は用いず、今の世代の探索結果のみを用いるから。
			// TODO
			//     : depth が異なっても詰みならentry を返しちゃうと、
			//       例えばdepth=6 なら詰みだったけど、depth=10だとdepth制限で引き分けになってしまう時、
			//       depth=6 で詰みだったのに、最大手数引き分けで不詰み(dn=inf) に書き換えられてしまう。
			//     : ただ、root node ではこれが起きないはず。何故ならdepth が異なっても返すのは詰みが証明されている場合のみであり、
			//       root node で詰みが証明されればほぼほぼ直ちに探索は終了するはずだからである。
			//       -> なので、起きないはずは言い過ぎだけど、確率はより低いと言える。
			// TODO
			//     : ここから返る場合、もしdepth が異なるなら書き換えるべきでない。。。？
			// 優越関係を満たす局面に証明済みの局面がある場合、それを返す
			if (entry.pn == 0) {
				if (hand.isEqualOrSuperior(entry.hand) && entry.num_searched != REPEAT) {
					// debug
					if (entry.hash_high == 2900215811ULL) {
						std::stringstream ss;
						ss << "[LNotDepth:" << entry.pn << "," << entry.dn
							<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "," << i << "]";
						enqueue_debug_str(ss.str());
					}
					entry.unlock();
					return entry;
				}
			}
			else if (entry.dn == 0) {
				if (entry.hand.isEqualOrSuperior(hand) && entry.num_searched != REPEAT) {
					entry.unlock();
					return entry;
				}
			}
			// TODO: 何故以下の場合に最大値を更新するのか？
			else if (entry.hand.isEqualOrSuperior(hand)) {
				if (entry.pn > max_pn) {
					max_pn = entry.pn;
				}
			}
			else if (hand.isEqualOrSuperior(entry.hand)) {
				if (entry.dn > max_dn) {
					max_dn = entry.dn;
				}
			}
		}
		entry.unlock();
	}

	// TODO
	//     : 
	// NOTE
	//     : 探索数が一番少ない局面程、hash に貯めておく価値も低いので、探索数が一番少ないentry を見つけて潰す。
	//       その際に、詰みが証明できていない局面から優先的に潰すようにする。
	//cout << "hash entry full" << endl;
	// 合致するエントリが見つからなかったので
	// 古いエントリをつぶす
	TTEntry* best_entry = nullptr;
	uint32_t best_num_searched = UINT32_MAX;         // NOTE: num_searched の最小値？
	TTEntry* best_entry_include_mate = nullptr;
	uint32_t best_num_searched_include_mate = UINT32_MAX;    // NOTE
	//     : 詰み局面のみのnum_searched 最小値？
	//       名前的には違うっぽいけど。。。

// NOTE
//     : 現状上書きする可能性のあるentry のlock は解放すべきでないはず。
//       一旦解放して、その間に価値のある別の情報が書き込まれたときにそれを潰すことになる。
//       まぁ十分大きな置換表があれば問題ないはず。
//       ただ、30スレッドぐらいで探索するのに、PUCTで64GBぐらいは食うはずで残りの64GBで十分大きいかというのは疑問が残る。
	for (auto& entry : entries.entries) {
		entry.lock();
		if (entry.pn != 0) {
			if (best_num_searched > entry.num_searched) {
				if (best_entry != nullptr) {
					best_entry->unlock();    // 書き換える可能性は無くなったので、一つ前のbestは解放
				}
				best_entry = &entry;
				best_num_searched = entry.num_searched;
			}
			else {
				entry.unlock();
			}
		}
		else {
			if (best_num_searched_include_mate > entry.num_searched) {
				if (best_entry_include_mate != nullptr) {
					best_entry_include_mate->unlock();    // 書き換える可能性は無くなったので、一つ前のbestは解放
				}
				best_entry_include_mate = &entry;
				best_num_searched_include_mate = entry.num_searched;
			}
			else {
				entry.unlock();
			}
		}
	}
	// NOTE
	//     : 詰みを証明できている局面の価値は高いので、
	//       詰みが証明できていない局面が一つも無い時にのみ、詰みが証明できている局面を潰す。
	if (best_entry == nullptr) {
		best_entry = best_entry_include_mate;
	}
	else {
		// best_entry_include_mate は、もしbest_entry として使われなければ一生解放されないのでここで解放
		if (best_entry_include_mate != nullptr) {
			best_entry_include_mate->unlock();
		}
	}

	best_entry->hash_high = hash_high;
	best_entry->hand = hand;
	best_entry->depth = depth;
	best_entry->pn = 1;
	best_entry->dn = 1;
	best_entry->generation = generation;
	best_entry->num_searched = 0;

	best_entry->unlock();
	return *best_entry;
}

template <bool or_node>
TTEntry& TranspositionTable::LookUp(const Position& n) {
	// NOTE: このLookUp<>() はLookUp() のラッパー
	auto& retVal = LookUp(n.getBoardKey(), or_node ? n.hand(n.turn()) : n.hand(oppositeColor(n.turn())), n.gamePly());
	return retVal;
}

// TODO
//     : lock 関連どうするか。
//       -> entries に書き込むことは無いはずで、あくまでアドレスを読み取るだけであり、探索中にアドレスが書き換えられる事は無いはずで、
//          クリティカルセクションにはなり得ないはず。 
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
			if (to_pc != Empty) {    // NOTE: move is capture
				const PieceType pt = pieceToPieceType(to_pc);
				hand.plusOne(pieceTypeToHandPiece(pt));
			}
		}
	}
	else {
		hand = n.hand(oppositeColor(n.turn()));
	}
	Key key = n.getBoardKeyAfter(move);



	// TODO
	//     : これデータ競合ある？
	//     : まず、Cluster tt[]; である。
	//       tt のCluster を読むのと、entry.pn = 0 とかするのは、排他制御されていない。(entry.lock() はtt からCluster を読むのに対して効果を発揮しない)
	//       
	entries = &tt[key & clusters_mask];
	hash_high = key >> 32;
	//return key;
}

// TODO
//     : lock 関連どうするか。
//       LookUpDirect 呼んじゃってるので、auto&& retVal = LookUpDirec() として前後を囲うべき？
//       てかあれか、これするにはn.getBoardKeyAfter(move); なるこのノードのkey が必要。
//       GetChildFirstEntry() から返してもらうか？
// moveを指した後の子ノードの置換表エントリを返す
template <bool or_node>
TTEntry& TranspositionTable::LookUpChildEntry(const Position& n, const Move move) {
	Cluster* entries;
	uint32_t hash_high;
	Hand hand;
	//const auto&& key = GetChildFirstEntry<or_node>(n, move, entries, hash_high, hand);
	GetChildFirstEntry<or_node>(n, move, entries, hash_high, hand);

	//std::cout << "key = [" << u64(key) << "]" << std::endl;
	return LookUpDirect(*entries, hash_high, hand, n.gamePly() + 1);
}

// TODO: やねうら王みたいに、最初から物理メモリ上に確保したい
// 領域を再確保することでreset してる。非推奨。
void TranspositionTable::Reset() {
	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}

	// NOTE
	//     : 特に型は指定せずに、ただただ(new_num_clusters * sizeof(Cluster) + CacheLineSize) * (1) byte の領域を確保
	// TODO
	//     : CacheLineSize とは？
	//     : ((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1)) って何してる？
	tt_raw = std::calloc(num_clusters * sizeof(Cluster) + CacheLineSize, 1);
	tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));
	clusters_mask = num_clusters - 1;
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
	// TOOD
	//     : これ、普通にif で判定すべきな気はする。
	//       現状、hash_size_mb = 512mb 未満だとアウト。恐らくね。
	assert(num_clusters >= POS_MUTEX_NUM && "expected num_clusters >= POS_MUTEX_NUM, but got num_clusters < POS_MUTEX_NUM");

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
	if (tt_raw == nullptr) {
		// なんかbad_alloc 出なかったのでこれで。
		std::cout << "Error: bad_alloc! Resize() failed to calloc" << std::endl;
		exit(1);
	}

	tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));
	clusters_mask = num_clusters - 1;
}

void TranspositionTable::NewSearch() {
	++generation;
	// TODO: なぜこうするの？初めはgeneration = -1 ってことかい？
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

// TODO
//     : 何故この実装で良いのか分からん。
//       別に、これ以上に持ち駒を増やしても詰まないって保証は無くないか？
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

// TODO
//     : lock関連どうすんねん。
//       LookUp() で貰ったentry にバチコリ書き込んますがな。position n のkey でposition mutex でもします？
//       
// @arg or_node
//     : OrNode である。
// @arg shared_stop
//     : 停止に_shared_stop を使う
// @arg thpn, thdn
//     : 一番初めのroot ではinf が渡されるっぽい。
//     : この局面に入ってくる直前において、
//       Position n の兄弟局面 の内、2番目に良い局面のpn, dn
//       ※2番目に良い局面とは、
//         OrNode なら2番目にpn が小さい局面、
//         AndNode なら2番目にdn が小さい局面 のことを指す。
template <bool or_node, bool shared_stop>
void ParallelDfPn::dfpn_inner(
	Position& n, const int thpn, const int thdn/*, bool inc_flag*/, const uint16_t maxDepth, int64_t& searchedNode, const int threadid
) {
	auto& entry = transposition_table->LookUp<or_node>(n);

	//if (threadid == 23) {
	//	std::stringstream ss;
	//	ss << "S";
	//	enqueue_inf_debug_str(ss.str());
	//}

	if (or_node) {
		// NOTE: 深さ制限に達したら、不詰みと判定
		if (n.gamePly() + 1 > maxDepth) {
			entry.lock();
			// NOTE
			//     : これは、depth が異なっても、持ち駒がある関係を満たせばentry を返してしまうので、
			//       詰み局面を最大手数引き分けによる不詰みで上書きしてしまうのを防ぐ。
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				// TODO20231026: 上書きのみ
				// NOTE: この場合、反証駒も何もない。だって最大手数引き分けなんだもの。
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				entry.num_searched = REPEAT;    // TODO: REPEAT って何？

			}
			entry.unlock();
			return;    // NOTE: serach_result = 中断(depth limit)
		}
	}

	// NOTE
	//     : hash が完全に一致しているのに盤面が違うことは無いと仮定した場合、
	//       pn=inf に書き換えられるバグが起きるのは、局面は同じなのに手数によって詰み不詰みが変わる場合。
	//       以下の場合は、手数によらずemtpy() であるので問題ない。
	// if (n is a terminal node) { handle n and return; }
	MovePicker<or_node> move_picker(n);
	if (move_picker.empty()) {    // NOTE: 手がこれ以上続かず詰ませられ無い or 詰まされたので投了するしかない
		// nが先端ノード
		entry.lock();

		if (or_node) {

			// 自分の手番でここに到達した場合は王手の手が無かった、
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				// TODO20231026: 上書きのみ
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				// 反証駒
				// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
				entry.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
			}
		}
		else {
#ifdef DFPN_MOD_V3
			if (entry.dn != 0) {
#else
			if (true) {
#endif
				// 相手の手番でここに到達した場合は王手回避の手が無かった、
				// 1手詰めを行っているため、ここに到達することはない
				// debug
				if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23) {
					std::stringstream ss;
					ss << "[pn:" << entry.pn << "," << entry.dn
						<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "]";
					enqueue_debug_str(ss.str());
				}
				// TODO20231026: 上書きのみ
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
			}
		}
		entry.unlock();
		return;    // NOTE: serach_result = 不詰み || 詰み
	}

	//// NOTE
	////     : 一旦、2 or 3手詰め 探索をする。
	// 新規節点で固定深さの探索を併用
	entry.lock();
	if (entry.num_searched == 0) {
		entry.unlock();

		if (or_node) {
			// 3手詰みチェック
			Color us = n.turn();
			Color them = oppositeColor(us);

			StateInfo si;
			StateInfo si2;

			const CheckInfo ci(n);
			for (const auto& ml : move_picker) {    // NOTE: OrNode での指し手を探索
				const Move& m = ml.move;

				n.doMove(m, si, ci, true);

				// 千日手のチェック
				if (n.isDraw(16) == RepetitionWin) {
					// NOTE: 連続王手の千日手
					// 受け側の反則勝ち
					n.undoMove(m);
					continue;
				}

				// TODO
				//     : 途中でentry2 の値が書き換えられた場合に、何か不都合あるかね？
				//       -> あー、あれだ、情報を書き換える時に全部の情報を書き換える訳じゃないから、
				//          途中で置換された場合、基本的にはhash_high & depth vs その他の情報 で異なる局面の情報が一つのentry に書き込まれてしまう。
				//          それで言うと、root node のentry もそうなる。ずっとlock しないといけなくなる。
				//          これ、一手or二手進めた時に、root と全く同じkey になる可能性ってあるかな？
				//          因みに、4手進めたら千日手で同じkey どころか同じ局面になり得るのでデッドロックになるのでダメ。
				auto& entry2 = transposition_table->LookUp<false>(n);

				// この局面ですべてのevasionを試す
				MovePicker<false> move_picker2(n);

				if (move_picker2.size() == 0) {
					// NOTE: 後手に逃げる手が無いので、今回の指し手m は詰ませる手である。
					// 1手で詰んだ
					n.undoMove(m);

					// TODO
					//     : kInfinitePnDn + 1 と+1 する理由は何？
					//       -> get_pv_inner とかでも使われてて、使われてるところ見た感じ、
					//          AndNode では+1, OrNode では+2 っぽい。。。?
					//       -> もっと言うと、+1 の時、その局面は詰み局面そのもの？
					//          ほんでもって+2 の時、1手詰め局面かな？

					// NOTE
					//     : hash_high は一度書き込まれたら、新しい世代or置換 されない限りは書き込まれない。
					//       なので、レアだし、debug 専用だから一旦lock は無しで。
					entry2.lock();
#ifdef DFPN_MOD_V3
					if (entry2.dn != 0) {
#else
					if (true) {
#endif
						// debug
						if (entry2.hash_high == DEBUG_HASH_HIGH_TAG && entry2.depth == 23) {
							std::stringstream ss;
							ss << "[pn:" << entry2.pn << "," << entry2.dn
								<< "," << entry2.depth << "," << entry2.hand.value() << "," << entry2.generation << "]";
							enqueue_debug_str(ss.str());
						}
						// TODO20231026: 上書きのみ
						entry2.pn = 0;
						entry2.dn = kInfinitePnDn + 1;
					}
					entry2.unlock();
					entry.lock();
#ifdef DFPN_MOD_V3
					// TODO20231026
					//     : dn = 0の場合は上書きしても良いのでは？
					//       だって、最大手数で不詰み判定された局面が、より短手数で発見できたら詰みに変えれる訳で。。
					//       -> 逆にダメな場合ってなんかある？
					if (entry.dn != 0) {
#else
					if (true) {
#endif
						// debug
						if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23) {
							std::stringstream ss;
							ss << "[pn:" << entry.pn << "," << entry.dn
								<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "]";
							enqueue_debug_str(ss.str());
						}
						// TODO20231026: 上書きのみ
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
					}
					entry.unlock();
					return;    // NOTE: serach_result = 詰み
				}

				// TODO
				//     : 何故+2 ？
				//     : move_picker より先にコッチチェックすべきでは？？
				// NOTE: 手数限界を突破したので、不詰み扱い
				if (n.gamePly() + 2 > maxDepth) {
					n.undoMove(m);

					entry2.lock();
#ifdef DFPN_MOD_V2
					if (entry2.pn != 0) {
#else
					if (true) {
#endif
						// TODO20231026: 上書きのみ
						entry2.pn = kInfinitePnDn;
						entry2.dn = 0;
						entry2.num_searched = REPEAT;
					}
					entry2.unlock();

					continue;
				}

				// NOTE
				//     : 今は遷移してAndNode に居るので、一つでも詰みを逃れる指し手があれば、NEXT_CHECK へとジャンプする。
				const CheckInfo ci2(n);
				for (const auto& move : move_picker2) {    // NOTE: AndNode での指し手を探索
					const Move& m2 = move.move;

					// この指し手で逆王手になるなら、不詰めとして扱う
					if (n.moveGivesCheck(m2, ci2))
						goto NEXT_CHECK;

					n.doMove(m2, si2, ci2, false);

					const auto tmpMove = n.mateMoveIn1Ply<mmin1ply_Additional>();
					if (tmpMove) {
						auto& entry1 = transposition_table->LookUp<true>(n);
						entry1.lock();
#ifdef DFPN_MOD_V3
						if (entry1.dn != 0) {
#else
						if (true) {
#endif
							// TODO: 証明駒をset すべき。
							// debug
							if (entry1.hash_high == DEBUG_HASH_HIGH_TAG && entry1.depth == 23) {
								std::stringstream ss;
								std::ostringstream oss_pos;
								n.print(oss_pos);
								ss << "[Or:" << tmpMove.toUSI() << "," << entry1.pn << "," << entry1.dn
									<< "," << entry1.depth << "," << entry1.hand.value() << "," << entry1.generation << "\n" << oss_pos.str() << "]";
								enqueue_debug_str(ss.str());
							}

							// TODO20231026
							//     : 上書きのみ
							//       ここでは、情報の参照はせず、この局面が詰む(特に1手詰めである)ことをentry に保存したい。
							//       -> つまり、depth とhand 一致、若しくは新しいentry に書き込む事のみが許される。
							entry1.pn = 0;
							entry1.dn = kInfinitePnDn + 2;

						}
						entry1.unlock();
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
				entry2.lock();
#ifdef DFPN_MOD_V3
				if (entry2.dn != 0) {
#else
				if (true) {
#endif
					// debug
					if (entry2.hash_high == DEBUG_HASH_HIGH_TAG && entry2.depth == 23) {
						std::stringstream ss;
						ss << "[pn:" << entry2.pn << "," << entry2.dn
							<< "," << entry2.depth << "," << entry2.hand.value() << "," << entry2.generation << "]";
						enqueue_debug_str(ss.str());
					}
					// TODO20231026: 上書きのみ
					entry2.pn = 0;
					entry2.dn = kInfinitePnDn;
				}
				entry2.unlock();
				entry.lock();
#ifdef DFPN_MOD_V3
				if (entry.dn != 0) {
#else
				if (true) {
#endif
					// debug
					if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23) {
						std::stringstream ss;
						ss << "[pn:" << entry.pn << "," << entry.dn
							<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "]";
						enqueue_debug_str(ss.str());
					}
					// TODO20231026: 上書きのみ
					entry.pn = 0;
					entry.dn = kInfinitePnDn;
				}
				entry.unlock();

				return;

				// NOTE
				//     : 今回探索したOrNode の指し手は、後手になんらかの逃げ道があった
			NEXT_CHECK:;
				n.undoMove(m);
				entry2.lock();
				if (entry2.num_searched == 0) {
					// TODO
					//     : 何故pn, dn が下記の値になる？
					//       pn はまぁ分かる気もするが、dn は何故?
					//       いや、あれか、AndNode だから、pn = move_picker2.size() なのは良いか。(ただこれ、子ノード展開しないといけないのでは？)
					//       だけど、dnは dn = 1; とすべきじゃないの？普通にするならさ？
					//     : そもそも、何故num_searched == 0 の時だけ？
			    	// TODO20231026: 上書きのみ
					entry2.num_searched = 1;
					entry2.pn = static_cast<int>(move_picker2.size());
					entry2.dn = static_cast<int>(move_picker2.size());
				}
				entry2.unlock();
			}
		}
		else {
			// 2手読みチェック
			StateInfo si2;
			// この局面ですべてのevasionを試す
			const CheckInfo ci2(n);
			for (const auto& move : move_picker) {    // NOTE: AndNode の指し手で、全部調べる。
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
				if (const Move move = n.mateMoveIn1Ply<mmin1ply_Additional>()) {
					auto& entry1 = transposition_table->LookUp<true>(n);

					entry1.lock();
#ifdef DFPN_MOD_V3
					if (entry1.dn != 0) {
#else
					if (true) {
#endif
						// debug
						if (entry1.hash_high == DEBUG_HASH_HIGH_TAG && entry1.depth == 23) {
							std::stringstream ss;
							std::ostringstream oss_pos;
							n.print(oss_pos);
							ss << "[And:" << move.toUSI() << "," << entry1.pn << "," << entry1.dn
								<< "," << entry1.depth << "," << entry1.hand.value() << "," << entry1.generation << "\n" << oss_pos.str() << "]";
							enqueue_debug_str(ss.str());
						}

						// TODO20231026: 上書きのみ
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
					entry1.unlock();
				}
				else {
					// 詰んでないので、m2で詰みを逃れている。
					// 不詰みチェック
					// 王手がない場合
					MovePicker<true> move_picker2(n);
					if (move_picker2.empty()) {
						auto& entry1 = transposition_table->LookUp<true>(n);
						entry1.lock();
#ifdef DFPN_MOD_V2
						if (entry1.pn != 0) {
#else
						if (true) {
#endif
							// TODO20231026: 上書きのみ
							entry1.pn = kInfinitePnDn;
							entry1.dn = 0;
							// 反証駒
							// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
							entry1.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
						}

						n.undoMove(m2);    // TODO: undoMove() 重いならクリティカルセクションからどかす。
						entry.lock();
#ifdef DFPN_MOD_V2
						if (entry.pn != 0) {
#else
						if (true) {
#endif
							// TODO20231026: 上書きのみ
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
						}
						entry.unlock();    // TODO: ここのunlock も要らん気もするな。
						entry1.unlock();
						return;
					}
					n.undoMove(m2);
					goto NO_MATE;
				}

				n.undoMove(m2);
			}

			// すべて詰んだ
			entry.lock();
#ifdef DFPN_MOD_V3
			if (entry.dn != 0) {
#else
			if (true) {
#endif
				// debug
				if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23) {
					std::stringstream ss;
					ss << "[pn:" << entry.pn << "," << entry.dn
						<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "]";
					enqueue_debug_str(ss.str());
				}
				// TODO20231026: 上書きのみ
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
			}
			entry.unlock();
			return;

		NO_MATE:;
		}

	}
	else {
		entry.unlock();
	}

	// TODO
	//     : どうしてこのタイミングで千日手判定するんや？
	//       -> 恐らくやけど、dfpn 呼ぶ局面では詰みの方が珍しいからやないかな。
	//          (先に短手数の詰みを確認しておいた方が、この関数を抜けると期待される時間は小さくなる。)
	//     : 引き分けなら評価値を書き換えるやつ、
	//       例えばdepth=6とdepth=9 で同じ局面が出てきた場合、depth=6 の詰み不詰みも千日手として値が書き換えられるが、
	//       qhapaq の記事のように、千日手は入り口から一番遠い局面で初めて不詰み扱いしなければならないので、
	//       depth=6 では不詰みとしてはいけない(はず)。「
	// 千日手のチェック
	// TODO20231026: 上書きのみ(以下のswitch 内全て)
	switch (n.isDraw(16)) {
	case RepetitionWin:
		// 連続王手の千日手による勝ち
		if (or_node) {
			// ここは通らないはず
			entry.lock();
#ifdef DFPN_MOD_V3
			if (entry.dn != 0) {
#else
			if (true) {
#endif
				// debug
				if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23) {
					std::stringstream ss;
					ss << "[pn:" << entry.pn << "," << entry.dn
						<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "]";
					enqueue_debug_str(ss.str());
				}
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
				entry.num_searched = REPEAT;
			}
			entry.unlock();
		}
		else {
			entry.lock();
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				entry.num_searched = REPEAT;
			}
			entry.unlock();
		}
		return;

	case RepetitionLose:
		// 連続王手の千日手による負け
		if (or_node) {
			entry.lock();
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				entry.num_searched = REPEAT;
			}
			entry.unlock();
		}
		else {
			// ここは通らないはず
			entry.lock();
#ifdef DFPN_MOD_V3
			if (entry.dn != 0) {
#else
			if (true) {
#endif
				// debug
				if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23) {
					std::stringstream ss;
					ss << "[pn:" << entry.pn << "," << entry.dn
						<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "]";
					enqueue_debug_str(ss.str());
				}
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
				entry.num_searched = REPEAT;
			}
			entry.unlock();
		}
		return;

	case RepetitionDraw:
		// 普通の千日手
		// ここは通らないはず
		entry.lock();
#ifdef DFPN_MOD_V2
		if (entry.pn != 0) {
#else
		if (true) {
#endif
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			entry.num_searched = REPEAT;
		}
		entry.unlock();
		return;

	case RepetitionSuperior:
		if (!or_node) {
			// NOTE
			//     : AndNode で優越局面になった場合、AndNode 視点で有利になってる訳で、
			//       OrNode 視点ではこの局面に遷移すべきでない。
			//       従って、OrNode 視点で"マイナスの評価値" となるようにpn = inf とする。
			// ANDノードで優越局面になっている場合、除外できる(ORノードで選択されなくなる)

			entry.lock();
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				entry.num_searched = REPEAT;
			}
			entry.unlock();
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
	//const auto&& debug_bk = n.getBoardKey();
	//const auto&& debug_hand_turn = n.hand(n.turn());
	for (const auto& move : move_picker) {
		auto& ttkey = ttkeys[&move - move_picker.begin()];

		transposition_table->GetChildFirstEntry<or_node>(n, move, ttkey.entries, ttkey.hash_high, ttkey.hand);

	}

	const Ply game_ply = n.gamePly();
	bool is_skip = false;
	// 定期的にroot に戻ってきて、また再度自身の担当depth へ現状の最良手順で辿っていく、のを繰り返すはず。。。
	if (threadid && (game_ply - 1) && game_ply <= threadid) {    // メインスレッドとルートノードはskip しない
		is_skip = true;
	}
	//if (threadid && (game_ply - 1)) {    // ダメ
	//	//is_skip = ((game_ply + skipPhase[threadid]) / skipSize[threadid]) % 3;    // ダメ
	//	is_skip = ((game_ply + skipPhase[threadid]) / skipSize[threadid]) % 2;
	//}
	//int node_count_down = (entry_num_searched / 10) + 1;    // ダメ
	//int node_count_down = my_max(10LL - entry_num_searched, 1);    // ダメ
	int node_count_down = 1;

	// NOTE
	//     : 任意のnode の探索数はmaxSearchNode を超えてはならない。
	while (searchedNode < maxSearchNode && !_should_stop<shared_stop>()) {
		//if (threadid == 23) {
		//	std::stringstream ss;
		//	ss << "W";
		//	enqueue_inf_debug_str(ss.str());
		//}

		// NOTE: 恐らくnum_searched について代入でない更新がなされるのはここだけ。
		entry.lock();
		++entry.num_searched;    // TODO: これだけの為にlock するのはもったいないよな。どっか移動したい。。。

		entry.unlock();

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
			uint32_t best_num_search = UINT32_MAX;    // NOTE: best_pn な子ノードのnum_searched 

#ifdef DFPN_MOD_V0
			int entry_pn = kInfinitePnDn;
			int entry_dn = 0;
#else
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
#endif
			// 子局面の反証駒の積集合
			u32 pawn = UINT_MAX;
			u32 lance = UINT_MAX;
			u32 knight = UINT_MAX;
			u32 silver = UINT_MAX;
			u32 gold = UINT_MAX;
			u32 bishop = UINT_MAX;
			u32 rook = UINT_MAX;
			bool repeat = false; // 最大手数チェック用    // TODO: これなんやねん。
			//int tmpCount = 0;    // debug
			for (const auto& move : move_picker) {
				auto& ttkey = ttkeys[&move - move_picker.begin()];
				// TODO
				//     : これlock するか、完全なコピーを取ってそいつを使うか、どっちが速いだろうか。テストすべきかな～？
				//       というか、途中で結果変わるの割と不都合な気もするな。知らんけど。
				//       _mtx はコピーしなくて良いから、struct TTEntryNoMtx を作って、そいつにコピー使用かしら。
				//const auto& child_entry = transposition_table->LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				// TODO20231026: 情報の参照のみ。
				auto& child_entry = transposition_table->LookUpDirect(
					*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1
				);    // 本当はconst auto&
				child_entry.lock();


				// debug
				entry.lock();
				if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23 && child_entry.pn == 0) {
					std::stringstream ss;
					ss << "[Corch:" << Move(move).toUSI() << "," << child_entry.pn << "," << child_entry.dn << "," << child_entry.hash_high
						<< "," << child_entry.depth << "," << child_entry.hand.value() << "," << child_entry.generation << "]";
					enqueue_debug_str(ss.str());
				}
				entry.unlock();

				// NOTE
				//     : ここでは、"depth が異なるが、優越関係を満たしたら返す"、がワンちゃん活きる。
				//       何故なら、LookUp したentry のpn, dn を(真に(pn == 0を詰みとして扱っている))使ってるからである。
				if (child_entry.pn == 0) {    // NOTE: OrNode で子ノードの一つが詰みだったので、現在のノードも詰み
					// 詰みの場合
#ifdef DFPN_MOD_V0
					entry_pn = 0;
					entry_dn = kInfinitePnDn;
#else
					entry.pn = 0;
					entry.dn = kInfinitePnDn;
#endif
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
						entry.lock();
						if (entry.hand.numOf(hp) > child_entry.hand.numOf(hp)) {
							// TODO20231026
							//     : 上の条件式で参照してるが、これは上書きのみと言えるんじゃないかな。。。？
							//       より汎用性を高められるかを確認してるだけだと思う。上の条件式はね。
							entry.hand = child_entry.hand;
							entry.hand.plusOne(move.move.handPieceDropped());    // TODO: 何故、これ、.plusOne(hp); にしない？
						}
						// TODO: このunlock、if else 全体にして、更に最後のunlock をなくせるはず。
						// (break で抜けたらすぐにまたentry.lock() するので。。。)
						entry.unlock();
					}
					// TODO
					//     : 駒を取る場合は、問答無用で今回の指し手についての証明駒を採用するっぽい。何故？？
					// 後手の駒を取る手ならば、証明駒から削除する
					else {
						const Piece to_pc = n.piece(move.move.to());
						if (to_pc != Empty) {    // NOTE: 駒を取る手である
							entry.lock();
							// TODO20231026: 上書きのみと言えそう。。。？
							entry.hand = child_entry.hand;
							const PieceType pt = pieceToPieceType(to_pc);
							const HandPiece hp = pieceTypeToHandPiece(pt);
							if (entry.hand.exists(hp)) {
								entry.hand.minusOne(hp);
							}
							entry.unlock();
						}
					}
					//cout << bitset<32>(entry.hand.value()) << endl;
					// NOTE
					//     : 今はOrNode 、つまり詰みについてOr であり、詰みが見つかったのでbreak
					child_entry.unlock();
					break;
				}
				// TODO
				//     : AndNode の時はこれに対応する処理が無いの、なぁぜなぁぜ？
				//       
				// NOTE
				//     : root node(entry) が不詰みなら基本ここを通る(はず)。
				//       (置換表から貰った局面(entry)では不詰みと言い切れたけどこの局面では違う場合(子ノードで詰んでる局面アリ)、上のif にい)
				//     : 以下のように、反証駒が登録されてる任意の子ノードの、任意の反証駒(ex 歩, 香, 桂馬, ...) についての最小値を求めれば、それが則ち積集合である。
				//     : true にする意味ない。何故なら、反証駒は不詰みの時のみ必要であり、不詰みとは言い切れないと判明した瞬間、
				//       わざわざ反証駒を計算する意味が無くなる。
				//       -> if (true) からif (entry_dn == 0) に。
#ifdef DFPN_MOD_V0
				else if (entry_dn == 0) {
#else
				else if (entry.dn == 0) {
#endif
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
				// NOTE
				//     : pn は最小値を選び、dn は合計値を取る
#ifdef DFPN_MOD_V0
				entry_pn = std::min(entry_pn, child_entry.pn);
				entry_dn += child_entry.dn;
#else
				entry.pn = std::min(entry.pn, child_entry.pn);
				entry.dn += child_entry.dn;
#endif

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
				child_entry.unlock();
			}    // NOTE: for 終了

			// TODO
			//     : kInfinitePnDn がどういう場面で代入されるか 把握できてない。
			//     : これは、子ノードに複数dn = kInfinitePnDn な奴が合った場合にclip するのが狙い？
			entry.lock();
#ifdef DFPN_MOD_V0
#ifdef DFPN_MOD_V2
			if (entry.pn != 0 && entry.dn != 0) {    // 書き込む値の両方が完全に未知の場合、元のpn, dn どちらもチェックする。
#else
			if (true) {
#endif
				// debug
				if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23) {
					std::stringstream ss;
					ss << "[Cor:" << entry_pn << "," << entry.pn << "," << entry.dn
						<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "]";
					enqueue_debug_str(ss.str());
				}
				// TODO20231026: 上書きのみ
				entry.pn = entry_pn;
				entry.dn = std::min(entry_dn, kInfinitePnDn);
			}
#else
			entry.dn = std::min(entry.dn, kInfinitePnDn);
#endif



			// NOTE
			//     : OrNode は不詰みに関してAnd なので、全部見てからでないとdn == 0 は途中では言い切れない。
			//       全部見終わったので、ここで初めて"dn == 0 なら不詰みである" が真となる。
			//     : df-pn の鉄則「詰み || 不詰み が判明したら、証明駒 || 反証駒 を計算して置換表にset」
			//     : この今の局面が不詰みである(entry_dn==0)から反証駒をset するのである。
			//       いつ何時勝手に変更されているか分からないentry.dn の値を用いてはならない。
			if (entry_dn == 0) {
				// 不詰みの場合
				//cout << n.hand(n.turn()).value() << "," << entry.hand.value() << ",";
				// NOTE
				//     : bool repeat = false; なるフラグについてのコメントや、以下のコメントなどを見るに、
				//       REPEAT とは最大手数で探索中断したことを示すフラグなのかもしれんな。
				// 最大手数で不詰みの局面が優越関係で使用されないようにする
				if (repeat)
					entry.num_searched = REPEAT;
				else {
					// TODO20231026
					//     : else if (pawn > curr_pawn) pawn = curr_pawn; ではないんだ？
					//       こうではなくて、持ち駒を拡大する方向に更新するんだ？なんで？
					// 先手が一枚も持っていない種類の先手の持ち駒を反証駒から削除する
					u32 curr_pawn = entry.hand.template exists<HPawn>(); if (curr_pawn == 0) pawn = 0; else if (pawn < curr_pawn) pawn = curr_pawn;
					u32 curr_lance = entry.hand.template exists<HLance>(); if (curr_lance == 0) lance = 0; else if (lance < curr_lance) lance = curr_lance;
					u32 curr_knight = entry.hand.template exists<HKnight>(); if (curr_knight == 0) knight = 0; else if (knight < curr_knight) knight = curr_knight;
					u32 curr_silver = entry.hand.template exists<HSilver>(); if (curr_silver == 0) silver = 0; else if (silver < curr_silver) silver = curr_silver;
					u32 curr_gold = entry.hand.template exists<HGold>(); if (curr_gold == 0) gold = 0; else if (gold < curr_gold) gold = curr_gold;
					u32 curr_bishop = entry.hand.template exists<HBishop>(); if (curr_bishop == 0) bishop = 0; else if (bishop < curr_bishop) bishop = curr_bishop;
					u32 curr_rook = entry.hand.template exists<HRook>(); if (curr_rook == 0) rook = 0; else if (rook < curr_rook) rook = curr_rook;
					// TODO20231026
					//     : ここは何？参照と上書き両方？それとも上書きのみ？
					// 反証駒に子局面の証明駒の積集合を設定
					entry.hand.set(pawn | lance | knight | silver | gold | bishop | rook);
					//cout << entry.hand.value() << endl;
				}
			}
			else {
				// NOTE
				//     : if (entry.pn >= thpn || entry.dn >= thdn) { break; } と評価しており、
				//       第二候補の兄弟局面のスコアを超過して初めて打ち切りたいので、"+1" する。
				if constexpr (EPS == 0) {    // 標準
					thpn_child = std::min(thpn, second_best_pn + 1);
				}
				else if constexpr (EPS > 0) {
					// より子ノードの長く滞在
					thpn_child = std::min(thpn, static_cast<int>(_EPS_PLUS_ONE * second_best_pn + 1));
				}
				else {
					// より子ノードに短く滞在
					thpn_child = std::min(
						thpn, best_pn + static_cast<int>(_EPS_PLUS_ONE * (second_best_pn - best_pn) + 1)
					);
				}
				thdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn);
			}
			// TODO
			//     : 一応、if (entry.dn == 0) の場合は最後にunlock() して、
			//       else の場合はthdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn); 
			//       だけを先にやってunlock() すればちょっと速くなるはず。。。
			//entry.unlock();    // NOTE: 閾値確認の終わりに移動
		}
		else {    // AndNode
			// ANDノードでは最も反証数の小さい = 王手の掛け方の少ない = 不詰みを示しやすいノードを選ぶ
			int best_dn = kInfinitePnDn;
			int second_best_dn = kInfinitePnDn;
			int best_pn = 0;
			uint32_t best_num_search = UINT_MAX;


			// NOTE: MOD_V0 を適用する際に、ここのentry.pn = 0, entry.dn = kInfinitePnDn を削除するのを忘れていた。
#ifdef DFPN_MOD_V0
			int entry_pn = 0;
			int entry_dn = kInfinitePnDn;
#else
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
#endif

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
				//const auto& child_entry = transposition_table->LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				// TODO20231026: 情報の参照のみ。
				auto& child_entry = transposition_table->LookUpDirect(
					*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1
				);    // 本当はconst auto&
				child_entry.lock();

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
#ifdef DFPN_MOD_V0
					entry_pn = kInfinitePnDn;
					entry_dn = 0;
#else
					entry.pn = kInfinitePnDn;
					entry.dn = 0;
#endif
					entry.lock();
					// 最大手数で不詰みの局面が優越関係で使用されないようにする
					if (child_entry.num_searched == REPEAT)
						entry.num_searched = REPEAT;
					else {
						// 子局面の反証駒を設定
						// 打つ手ならば、反証駒から削除する
						if (move.move.isDrop()) {
							const HandPiece hp = move.move.handPieceDropped();
							// TODO20231026: 恐らく上書きのみと言えるはず。
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
					// TDOO: entry はbreak 抜けたら直後に又lock() するから、ここで手放す必要はない。
					entry.unlock();
					child_entry.unlock();
					break;
				}

#ifdef DFPN_MOD_V0
				entry_pn += child_entry.pn;
				entry_dn = std::min(entry_dn, child_entry.dn);
#else
				entry.pn += child_entry.pn;
				entry.dn = std::min(entry.dn, child_entry.dn);
#endif

				if (child_entry.dn < best_dn ||
					child_entry.dn == best_dn && best_num_search > child_entry.num_searched) {
					second_best_dn = best_dn;
					best_dn = child_entry.dn;
					best_pn = child_entry.pn;
					if ((child_entry.pn >= kInfinitePnDn && child_entry.dn != 0) || (child_entry.dn >= kInfinitePnDn && child_entry.pn != 0)) {
						std::cout << "[PnDnError" << child_entry.pn << "," << child_entry.dn << "]";
					}
					best_move = move;
				}
				else if (child_entry.dn < second_best_dn) {
					second_best_dn = child_entry.dn;
				}
				child_entry.unlock();
			}    // NOTE: for 終了
			entry.lock();
#ifdef DFPN_MOD_V0
    #ifdef DFPN_MOD_V2
			if (entry.pn != 0 && entry.dn != 0) {
    #else
			if (true) {
    #endif
				// debug
				if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23) {
					std::stringstream ss;
					ss << "[Cand:" << entry_pn << "," << entry.pn << "," << entry.dn
						<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "]";
					enqueue_debug_str(ss.str());
				}
				// TODO20231026: 上書きのみ。
				entry.pn = std::min(entry_pn, kInfinitePnDn);
				entry.dn = entry_dn;
			}
#else
			entry.pn = std::min(entry.pn, kInfinitePnDn);
#endif

			// NOTE
			//     : ここで初めて"pn == 0 ならば詰みである" が真となる。
			//     : df-pn の鉄則より、反証駒の計算とset
			if (entry_pn == 0) {
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

				// TODO20231026
				//     : ここは何？参照と上書き両方？それとも上書きのみ？
				// NOTE
				//     : 進研ゼミでやったところ
				//     : ここはAndNode なので、setPP() するときは、受け側が一枚も持ってなくて、攻め手が独占してる駒を
				//       証明駒に追加しないといけない。つまり、them にoppositeColor のhand を渡すべきでは？
				// 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
				if (!(n.checkersBB() & n.attacksFrom<King>(n.kingSquare(n.turn())) || n.checkersBB() & n.attacksFrom<Knight>(n.turn(), n.kingSquare(n.turn())))) {
					entry.hand.setPP(n.hand(oppositeColor(n.turn())), n.hand(n.turn()));
				}
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
				//thdn_child = std::min(thdn, second_best_dn + 1);
				if constexpr (EPS == 0) {
					thdn_child = std::min(thdn, second_best_dn + 1);
				}
				else if constexpr (EPS > 0) {
					// より子ノードの長く滞在
					thdn_child = std::min(thdn, static_cast<int>(_EPS_PLUS_ONE * second_best_dn + 1));
				}
				else {
					// より子ノードに短く滞在
					thdn_child = std::min(
						thdn, best_dn + static_cast<int>(_EPS_PLUS_ONE * (second_best_dn - best_dn) + 1)
					);
				}
			}
			//entry.unlock();    // NOTE: 閾値確認の終わりに移動
		}
		// NOTE
		//     : 詰み(pn == 0) と分かった場合は、dn == inf となっているので、dn の制限でbreak する。
		// if (pn(n) >= thpn || dn(n) >= thdn) break; // termination condition is satisfied
		if (entry.pn >= thpn || entry.dn >= thdn || node_count_down == 0) {
			entry.unlock();
			break;
		}
		entry.unlock();

		StateInfo state_info;
		n.doMove(best_move, state_info);
		++searchedNode;
		if (is_skip) {
			// TODO
			//     : 深さに応じてこれ変えるとか、そもそもを3とか4にするとかね？色々ある。
			thpn_child = std::min(static_cast<int>(thpn_child * THPN_MULTIPLY), kInfinitePnDn);
			thdn_child = std::min(static_cast<int>(thdn_child * THDN_MULTIPLY), kInfinitePnDn);
		}
		dfpn_inner<!or_node, shared_stop>(n, thpn_child, thdn_child/*, inc_flag*/, maxDepth, searchedNode, threadid);
		n.undoMove(best_move);

		node_count_down -= is_skip;
		// NOTE
		//     : ここで抜けてはならないはず。抜けるのは、node制限で完全に探索を中止する時のみ。
		//       絶対に、此のノードに、子ノードから帰還してきた時は、このwhlie節の大部分を占める、pn, dn の更新作業が必要。
		//       この関数を並列で動かしても問題が無いのは、必ずpn, dn の更新を一つ上に上がるに行っているからである。
		//       一つでも更新をぬかすと、その瞬間反映されない子ノードでのpn, dn の更新 が出てきてしまう。
	}
}

// 詰みの手返す
Move ParallelDfPn::dfpn_move(Position & pos) {
	MovePicker<true> move_picker(pos);
	for (const auto& move : move_picker) {
		const auto& child_entry = transposition_table->LookUpChildEntry<true>(pos, move);
		if (child_entry.pn == 0) {
			return move;
		}
	}

	return Move::moveNone();
}

// FIXME: 永遠に循環して終わらないことがあるっぽい。。。？
template<bool or_node, bool safe>
int ParallelDfPn::get_pv_inner(Position & pos, std::vector<Move>&pv) {
	std::stringstream ss;
	pos.print(ss);
	if (or_node) {
		// ORノードで詰みが見つかったらその手を選ぶ
		MovePicker<true> move_picker(pos);
		for (const auto& move : move_picker) {
			// NOTE: コイツはlock, unlock 以外はconst
			auto& child_entry = transposition_table->LookUpChildEntry<true>(pos, move);
			if (safe) child_entry.lock();

#ifdef DEBUG_GETPV_20231026_0
			const auto debug_bk = pos.getBoardKey();
			// debug
			// ある局面では、ある子局面以外には遷移せず、ある子局面以外は全て飛ばす。
			if (debug_bk == DEBUG_HASH_P_5) {
				std::cout << "PvOr5:" << Move(move).toUSI() << "," << child_entry.hash_high
					<< "," << child_entry.pn << "," << child_entry.dn << "," << child_entry.depth << "," << child_entry.generation << std::endl;
			}
#endif

			// debug
			if (child_entry.hash_high == DEBUG_HASH_HIGH_0) {
				std::stringstream ss;
				std::ostringstream oss;
				pos.print(oss);
				ss << "[PvOr0:" << child_entry.pn << "," << child_entry.dn << "," << child_entry.depth << "," << child_entry.hand.value() << "," << child_entry.generation
					<< "\n" << oss.str() << "]";
				enqueue_debug_str(ss.str());
			}

			// debug
			if (child_entry.hash_high == DEBUG_HASH_HIGH_1) {
				std::stringstream ss;
				std::ostringstream oss;
				pos.print(oss);
				ss << "[PvOr1:" << child_entry.pn << "," << child_entry.dn << "," << child_entry.depth << "," << child_entry.hand.value() << "," << child_entry.generation
					<< "\n" << oss.str() << "]";
				enqueue_debug_str(ss.str());
			}

			if (child_entry.pn == 0) {
				if (child_entry.dn == kInfinitePnDn + 1) {
					if (safe) child_entry.unlock();
					//std::cout << "[Or][child_entry.dn == kInfinitePnDn + 1] " << Move(move).toUSI() << std::endl;
					pv.emplace_back(move);
					return 1;
				}
				if (safe) child_entry.unlock();

				StateInfo state_info;
				pos.doMove(move, state_info);
				const auto draw_type = pos.isDraw(16);
				switch (draw_type) {
					// NOTE: 以下のどちらかであれば再帰。
				case NotRepetition:
				case RepetitionSuperior:
				{
					//std::cout << "[Or][isDraw()] " << Move(move).toUSI() << std::endl;
					pv.emplace_back(move);
					const auto depth = get_pv_inner<false, safe>(pos, pv);
					pos.undoMove(move);
					return depth + 1;
				}
				default:
					break;
				}
				pos.undoMove(move);
			}
			else {
				if (safe) child_entry.unlock();
			}
		}
	}
	else {
		// ANDノードでは詰みまでが最大手数となる手を選ぶ
		int max_depth = 0;
		std::vector<Move> max_pv;    // NOTE: 最大のPV
		MovePicker<false> move_picker(pos);
		for (const auto& move : move_picker) {
			// NOTE: コイツはlock, unlock 以外はconst
			auto& child_entry = transposition_table->LookUpChildEntry<false>(pos, move);
			if (safe) child_entry.lock();

#ifdef DEBUG_GETPV_20231026_0
			const auto debug_bk = pos.getBoardKey();
			// debug
			// ある局面では、ある子局面以外には遷移せず、ある子局面以外は全て飛ばす。
			if (debug_bk == DEBUG_HASH_P_0 && child_entry.hash_high != DEBUG_HASH_HIGH_0) {
				continue;
			}
			else if (debug_bk == DEBUG_HASH_P_1) {
				std::cout << "PvAnd1:" << Move(move).toUSI() << "," << child_entry.hash_high
					<< "," << child_entry.pn << "," << child_entry.dn << "," << child_entry.depth << "," << child_entry.generation << std::endl;
			}
			else if (debug_bk == DEBUG_HASH_P_2_SKIP && child_entry.hash_high != DEBUG_HASH_HIGH_2_SKIP) {
				continue;
			}
			else if (debug_bk == DEBUG_HASH_P_3 && child_entry.hash_high != DEBUG_HASH_HIGH_3) {
				continue;
			}
			else if (debug_bk == DEBUG_HASH_P_4 && child_entry.hash_high != DEBUG_HASH_HIGH_4) {
				continue;
			}
			else if (debug_bk == DEBUG_HASH_P_6) {
				std::cout << "PvAnd6:" << Move(move).toUSI() << "," << child_entry.hash_high
					<< "," << child_entry.pn << "," << child_entry.dn << "," << child_entry.hand.value()
					<< "," << child_entry.depth << "," << child_entry.generation << std::endl;
			}
			if (debug_bk == DEBUG_HASH_P_1 && child_entry.hash_high != DEBUG_HASH_HIGH_1) {
				std::cout << "[PvAnd1: continue:" << Move(move).toUSI() << "]" << std::endl;
				continue;
			}
#endif

			//// debug
			// ある指し手によって遷移する局面の情報を表示する。(pn == 0 なら、この局面自体は問題ないということになる。)
			if (child_entry.hash_high == DEBUG_HASH_HIGH_0) {
				std::stringstream ss;
				std::ostringstream oss;
				pos.print(oss);
				ss << "[PvAnd0:" << child_entry.pn << "," << child_entry.dn << "," << child_entry.depth
					<< "," << child_entry.hand.value() << "," << child_entry.generation
					<< "\n" << oss.str() << "]";
				enqueue_debug_str(ss.str());
			}
			else if (child_entry.hash_high == DEBUG_HASH_HIGH_1) {
				std::stringstream ss;
				std::ostringstream oss;
				pos.print(oss);
				ss << "[PvAnd1:" << child_entry.pn << "," << child_entry.dn << "," << child_entry.depth
					<< "," << child_entry.hand.value() << "," << child_entry.generation
					<< "\n" << oss.str() << "]";
				enqueue_debug_str(ss.str());
			}
			else if (child_entry.hash_high == DEBUG_HASH_HIGH_3) {
				std::stringstream ss;
				std::ostringstream oss;
				pos.print(oss);
				ss << "[PvAnd3:" << child_entry.pn << "," << child_entry.dn << "," << child_entry.depth
					<< "," << child_entry.hand.value() << "," << child_entry.generation
					<< "\n" << oss.str() << "]";
				enqueue_debug_str(ss.str());
				std::cout << ss.str() << std::endl;
			}
			else if (child_entry.hash_high == DEBUG_HASH_HIGH_4) {
				std::stringstream ss;
				std::ostringstream oss;
				pos.print(oss);
				ss << "[PvAnd4:" << child_entry.pn << "," << child_entry.dn << "," << child_entry.depth
					<< "," << child_entry.hand.value() << "," << child_entry.generation
					<< "\n" << oss.str() << "]";
				enqueue_debug_str(ss.str());
				std::cout << ss.str() << std::endl;
			}

			// NOTE
			//     : .dn == 0 の場合を考えない。何故ならここに来るときに通ったノードがpn == 0であり、
			//       ここはAndNode なので、アルゴリズムが正しければここは当然全ての指し手が詰むはずで、
			//       そんなものチェックしない。
			if (child_entry.pn == 0) {
				std::vector<Move> tmp_pv{ move };
				StateInfo state_info;
				pos.doMove(move, state_info);
				int depth = -kInfinitePnDn;

				// debug
				if (child_entry.hash_high == DEBUG_HASH_HIGH_0) {
					std::stringstream ss;
					ss << "[PvAnd0:child_entry.pn == 0]";
					enqueue_debug_str(ss.str());
				}
				else if (child_entry.hash_high == DEBUG_HASH_HIGH_1) {
					std::stringstream ss;
					ss << "[PvAnd1:child_entry.pn == 0]";
					enqueue_debug_str(ss.str());
				}

				if (child_entry.dn == kInfinitePnDn + 2) {
					if (safe) child_entry.unlock();

					//std::cout << "[And][child_entry.dn == kInfinitePnDn + 2]" << Move(move).toUSI() << std::endl;
					depth = 1;
					if (!pos.inCheck()) {
						// 1手詰みチェック
						Move mate1ply = pos.mateMoveIn1Ply<mmin1ply_Additional>();
						if (mate1ply) {
							tmp_pv.emplace_back(mate1ply);
						}
					}
					else
						get_pv_inner<true, safe>(pos, tmp_pv);
				}
				else {
					if (safe) child_entry.unlock();

					//std::cout << "[And][before dfpn_inner()]" << Move(move).toUSI() << std::endl;
					depth = get_pv_inner<true, safe>(pos, tmp_pv);

					// debug
					if (safe) child_entry.lock();
					if (child_entry.hash_high == DEBUG_HASH_HIGH_0) {
						std::stringstream ss;
						ss << "[PvAnd0:" << depth << "," << max_depth << "]";
						enqueue_debug_str(ss.str());
					}
					else if (child_entry.hash_high == DEBUG_HASH_HIGH_1) {
						std::stringstream ss;
						std::stringstream ss_immediate;
						for (size_t i = 0; i < tmp_pv.size(); i++) {
							ss_immediate << " " << tmp_pv[i].toUSI();
						}
						std::cout << "[PvAnd1:" << depth << "," << max_depth
							<< "," << tmp_pv.size() << ",pv=" << ss_immediate.str() << "]" << std::endl;
						enqueue_debug_str(ss.str());
					}
					if (debug_bk == DEBUG_HASH_P_1) {
						std::cout << "[PvAnd1:" << depth << "," << max_depth << "," << Move(move).toUSI() << "]" << std::endl;
					}
					else if (debug_bk == DEBUG_HASH_P_3) {
						std::cout << "[PvAnd1:" << depth << "," << max_depth << "," << Move(move).toUSI() << "]" << std::endl;
					}
					if (safe) child_entry.unlock();
				}
				pos.undoMove(move);

				// NOTE: より長いPVが見つかれば更新
				if (depth > max_depth) {
					max_depth = depth;
					max_pv = std::move(tmp_pv);
					for (int i = 0; i < max_pv.size(); ++i) {
						const auto& tmpMove = max_pv[i];
						//std::cout << "[And] max_pv[" << i << "] = " << tmpMove.toUSI() << std::endl;
					}
				}
			}
			else {
				if (safe) child_entry.unlock();
			}
		}
		if (max_depth > 0) {
			std::copy(max_pv.begin(), max_pv.end(), std::back_inserter(pv));
			return max_depth + 1;
		}
	}
	return -kInfinitePnDn;
}

// NOTE
//     : safe なら、他の探索と被っても問題ない。
//     : この関数自体を並列で呼ぶことは想定していないが、問題なさそうな気はしてる。
// PVと詰みの手返す
template <bool safe>
std::tuple<std::string, int, Move> ParallelDfPn::get_pv(Position & pos) {

	//std::cout << "info: get_pv() start" << std::endl;

	flush_debug_str();

	std::vector<Move> pv;
	int depth = -1;
	if constexpr (safe) {
		depth = get_pv_inner<true, true>(pos, pv);
	}
	else {    // unsafe
		depth = get_pv_inner<true, false>(pos, pv);
	}
	if (pv.size() == 0) {
		pv.emplace_back(Move(0));
	}
	//std::cout << "info: get_pv_inner() done" << std::endl;
	const Move& best_move = pv[0];
	std::stringstream ss;

	ss << best_move.toUSI();
	for (size_t i = 1; i < pv.size(); i++)
		ss << " " << pv[i].toUSI();

	return std::make_tuple(ss.str(), depth, best_move);
}

// 詰将棋探索のエントリポイント
template<bool shared_stop>
bool ParallelDfPn::dfpn(Position & r, int64_t & searched_node, const int threadid) {
	// キャッシュの世代を進める

	//std::cout << "debug: DLSHOGI dfpn::dfpn" << std::endl;
	//std::cout << "info: " << get_option_str() << std::endl;

	searched_node = 0;    // NOTE: 探索ノード数をreset
	if (!r.inCheck()) {
		// 1手詰みチェック
		Move mate1ply = r.mateMoveIn1Ply<mmin1ply_Additional>();
		if (mate1ply) {
			auto& child_entry = transposition_table->LookUpChildEntry<true>(r, mate1ply);
			child_entry.lock();
#ifdef DFPN_MOD_V3
			if (child_entry.dn != 0) {    // 流石に大丈夫だと思うけどね。
#else
			if (true) {
#endif
				child_entry.pn = 0;
				child_entry.dn = kInfinitePnDn + 1;
			}
			child_entry.unlock();
			return true;
		}
	}
	dfpn_inner<true, shared_stop>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, std::min(r.gamePly() + kMaxDepth, draw_ply), searched_node, threadid);
	//sync_cout << "info: dfpn done, threadid = " << threadid << ", done, " << get_now_str() << sync_endl;

	auto&& entry = transposition_table->LookUp<true>(r);

	//cout << searched_node << endl;

	/*std::vector<Move> moves;
	std::unordered_set<Key> visited;
	dfs(true, r, moves, visited);
	for (Move& move : moves)
	cout << move.toUSI() << " ";
	cout << endl;*/


	entry.lock();
	
	//// debug
	//if (threadid == 23) {
	//	std::stringstream ss;
	//	ss << "[flush start:" << threadid << "]";
	//	enqueue_inf_debug_str(ss.str());
	//	flush_inf_debug_str();
	//}

	const auto&& retVal = (entry.pn == 0);
	//sync_cout << "info: dfpn done, threadid = " << threadid << ", retVal = " << bts(retVal) << std::endl;
	//std::cout
	//	<< "[key = " << r.getBoardKey()
	//	<< "][high = " << (r.getBoardKey() >> 32) << "]" << std::endl;
	//std::cout << IO_LOCK;
	//this->_print_entry_info(entry);
	//std::cout << IO_UNLOCK;
	entry.unlock();
	//std::string pv_str;
	//int depth;
	//Move bestmove;
	//std::tie(pv_str, depth, bestmove) = this->get_pv<true>(r);
	//std::cout << "info: depth = [" << depth << "]" << std::endl;
	//std::cout << "info: bestmove = [" << bestmove.toUSI() << "]" << std::endl;
	//std::cout << "info: pv_str = [" << pv_str << "]" << std::endl;
	//flush_global_debug_str();
	//std::cout << IO_UNLOCK;


	return retVal;
}

// 詰将棋探索のエントリポイント
template<bool shared_stop>
bool ParallelDfPn::dfpn_andnode(Position & r, int64_t & searched_node, const int threadid) {
	// 自玉に王手がかかっていること

	// キャッシュの世代を進める
	//transposition_table->NewSearch();

	searched_node = 0;
	dfpn_inner<false, shared_stop>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, std::min(r.gamePly() + kMaxDepth, draw_ply), searched_node, -1 /* dummy threadid */);
	auto& entry = transposition_table->LookUp<false>(r);

	entry.lock();
	const auto&& retVal = (entry.pn == 0);
	entry.unlock();

	return retVal;
}

template<bool or_node>
void ParallelDfPn::print_entry_info(Position & n) {
	TTEntry& entry = transposition_table->LookUp<or_node>(n);
	this->_print_entry_info(entry);
}

// NOTE
//     : https://pknight.hatenablog.com/entry/20090826/1251303641
//         : .h に宣言、.cpp に定義が書いてある場合、明示的インスタンス化が必要。
//         :  明示的インスタンス化をしない場合、分割コンパイルが原因で以下の事象が発生する。
//            .h をinclude してるファイルでは、宣言があるので一旦コンパイルが通り、
//            .cpp では、template の関数が使われていないのでインスタンス化されずにコンパイルされる。
//            以上より、.obj をlink する時に、どこにも定義が無い！という自体が発生し、LNK2001 となる。
//         : 明示的インスタンス化は、template と書く。template<> では無い。
template std::tuple<std::string, int, Move> ParallelDfPn::get_pv<true>(Position & pos);
template std::tuple<std::string, int, Move> ParallelDfPn::get_pv<false>(Position & pos);

template void ParallelDfPn::print_entry_info<true>(Position & n);
template void ParallelDfPn::print_entry_info<false>(Position & n);

template bool ParallelDfPn::dfpn<true>(Position& r, int64_t& searched_node, const int threadid);
template bool ParallelDfPn::dfpn<false>(Position& r, int64_t& searched_node, const int threadid);

template bool ParallelDfPn::dfpn_andnode<true>(Position& r, int64_t& searched_node, const int threadid);
template bool ParallelDfPn::dfpn_andnode<false>(Position& r, int64_t& searched_node, const int threadid);

#endif    // DFPN_PARALLEL4