#include <unordered_set>

#include "my_dfpn_parallel3.hpp"



#ifdef DFPN_PARALLEL3

//// debug
// HASH_HIGH_DEBGU_TARGET は1番目にlock, unlock を実装したセクションでのターゲット
//constexpr int64_t HASH_HIGH_DEBGU_TARGET = 2155002330LL;
//constexpr int64_t HASH_HIGH_DEBGU_TARGET = 1422904718LL;
//constexpr int64_t HASH_HIGH_DEBGU_TARGET = 1128964956LL;
//constexpr int64_t HASH_HIGH_DEBGU_TARGET = 5581626913051496642LL >> 32;
constexpr uint64_t HASH_HIGH_DEBGU_TARGET = 0;    // NOTE: 0の局面なんてありえへんやろww ってことで0 にして無効化してるけど実際は知らん。

// HASH_HIGH_DEBGU_TARGET は2番目にlock, unlock を実装したセクションでのターゲット
//constexpr int64_t HASH_HIGH_DEBGU_TARGET2 = 1128964956LL;
//constexpr int64_t HASH_HIGH_DEBGU_TARGET2 = 5581626913051496642LL >> 32;
constexpr uint64_t HASH_HIGH_DEBGU_TARGET2 = 0;

// hash_high からboardKey 全体を知る為にLookUp に接地する奴
//constexpr int64_t HASH_HIGH_DEBGU_TARGET3 = 4112105184;
//constexpr int64_t HASH_HIGH_DEBGU_TARGET3 = -1;


//constexpr int64_t HASH_DEBUG_TARGET = 5581626913051496642LL;
//constexpr int64_t HASH_DEBUG_TARGET = 971728613859025696LL;
constexpr uint64_t HASH_DEBUG_TARGET = 0;

constexpr uint64_t HASH_DEBUG_TARGET2 = 17661357284713969480LL;


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
int64_t ParallelDfPn::HASH_SIZE_MB = 2048;
int ParallelDfPn::draw_ply = INT_MAX;
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
#ifdef DEBUG
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
		//assert(mtx == true && );
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
	auto& entries = tt[key & clusters_mask];
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

	//// debug
	//if (16721636401499889413LL >> 32 == hash_high) {
	//	int tmpCount = 0;
	//	for (size_t i = 0; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
	//		TTEntry& entry = entries.entries[i];
	//		if (generation == entry.generation) {
	//			++tmpCount;
	//		}
	//	}
	//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
	//	std::cout << "[16721636401499889413][LookUpDirect()][start] tmpCount = " << tmpCount << std::endl;
	//}

	// TODO
	//     : lock 獲得してる時間が長くて且つ情報の参照しかしないならコピーした方が良いけど、情報の参照しかしない保証はない。

	// 検索条件に合致するエントリを返す
	for (size_t i = 0; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
		TTEntry& entry = entries.entries[i];
		// debug
#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
			std::cout << "[LookUpDirect()][entry] lock" << std::endl;
		}
#endif
		entry.lock();

		//// debug
		//if (16721636401499889413LL >> 32 == entry.hash_high) {
		//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
		//	std::cout << "[16721636401499889413][LookUpDirect()][" << bts(generation != entry.generation) << "][" << i << "] (" << entry.pn << ", " << entry.dn << ") " << depth << ", " << hand.value() << std::endl;
		//}

		// TODO
		//     : break 又はreturn で抜ける直前では必ずunlock()
		//       それ以外は使わなくなったらunlock();
		//     : どっかunlock() されてなかったら、数百万もテストすればどっかはデッドロック起こしてくれるでしょ。
		//       その時に、CAS lock の所で待ち時間長い場合に警告出せばよい。
		// NOTE
		//     : 今の世代に一致しないものは、ゴミと見なす？(利用せずに上書きする？)
		if (generation != entry.generation) {

			//// debug
			//if (16721636401499889413LL >> 32 == hash_high) {
			//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
			//	std::cout << "[16721636401499889413][LookUpDirect()][new gen][" << i << "] (" << max_pn << ", " << max_dn << ") " << depth << ", " << hand.value() << std::endl;
			//}

#if 0
			if (HASH_DEBUG_TARGET >> 32 == hash_high) {
				std::stringstream ss;
				ss
					<< "[HASH_DEBUG_TARGET][LookUpDirect()][new gen][" << i << "] ("
					<< max_pn << ", " << max_dn << ") " << depth << ", " << hand.value();
				enqueue_global_debug_str(ss);
			}
#endif

#ifdef DEBUG_DEADLOCK_20231014_0
			if (hash_high == HASH_DEBUG_TARGET2 >> 32) {
				std::cout << "[HASH_DEBUG_TARGET2][LookUpDirect()][new] " << depth << ", " << max_pn << ", " << max_dn << std::endl;
			} 
#endif

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

			// debug
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
				std::cout << "[LookUpDirect()][entry][diff gen] unlock" << std::endl;
			}
#endif
			entry.unlock();
			return entry;
		}

		// TODO
		//     : hash と世代が一致するのは絶対に必要？ generation が進むタイミングは？(see dlshogi)
		if (hash_high == entry.hash_high && generation == entry.generation) {    // NOTE: 同じ局面と見なす必要条件

#ifdef DEBUG_DEADLOCK_20231014_0_
			if (entry.hash_high == HASH_DEBUG_TARGET2 >> 32) {
				std::cout << "[HASH_DEBUG_TARGET2] " << entry.depth << ", " << entry.pn << ", " << entry.dn << std::endl;
			}
#endif

			if (hand == entry.hand && depth == entry.depth) {
				// debug
#ifdef DEBUG
				if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
					std::cout << "[LookUpDirect()][entry][if (hand == entry.hand && depth == entry.depth)] unlock" << std::endl;
				}
#endif
				entry.unlock();
				// keyが合致するエントリを見つけた場合
				// 残りのエントリに優越関係を満たす局面があり証明済みの場合、それを返す
				for (i++; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
					TTEntry& entry_rest = entries.entries[i];
					// debug
#ifdef DEBUG
					if (entry_rest.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[LookUpDirect()][entry_rest] lock" << std::endl;
					}
#endif
					entry_rest.lock();
					if (generation != entry_rest.generation) {    // TODO: これcontinue じゃダメなんですか？
						entry_rest.unlock();
						break;
					}
					// TODO
					//     : ここから返る場合、もしdepth が異なるなら書き換えるべきでない。。。？
					if (hash_high == entry_rest.hash_high) {
						// NOTE
						//     : 詰ます側が、残りのentry の持ち駒以上に持ち駒を持っているなら上位互換で、詰むはず、的な話？
						//       -> key でアクセスしてるので、盤面だけで見ても一致するとは限らんよな？そのあたりどうなってん？
						if (entry_rest.pn == 0) {    // NOTE:  詰みが証明済み
							if (hand.isEqualOrSuperior(entry_rest.hand) && entry_rest.num_searched != REPEAT) {
#ifdef DEBUG
								if (entry_rest.hash_high == HASH_HIGH_DEBGU_TARGET) {
									std::cout << "[LookUpDirect()] unlock" << std::endl;
								}
#endif

#ifdef DEBUG_DEADLOCK_20231014_0
								if (entry_rest.hash_high == HASH_DEBUG_TARGET2 >> 32) {
									std::cout
										<< "[HASH_DEBUG_TARGET2][LookUpDirect()][entry_rest] "
										<< entry_rest.depth << ", " << entry_rest.pn << ", " << entry_rest.dn << std::endl;
								}
#endif
								entry_rest.unlock();
								return entry_rest;
							}
						}
						else if (entry_rest.dn == 0) {    // NOTE: 不詰み証明済み
							if (entry_rest.hand.isEqualOrSuperior(hand) && entry_rest.num_searched != REPEAT) {
#ifdef DEBUG
								if (entry_rest.hash_high == HASH_HIGH_DEBGU_TARGET) {
									std::cout << "[LookUpDirect()] unlock" << std::endl;
								}
#endif
								entry_rest.unlock();
								return entry_rest;
							}
						}
					}
#ifdef DEBUG
					if (entry_rest.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[LookUpDirect()] unlock" << std::endl;
					}
#endif
					entry_rest.unlock();
				}

#ifdef DEBUG
				if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
					std::cout << "[LookUpDirect()] return" << std::endl;
				}
#endif
				// NOTE: より強いことが言えるentry が無ければ、先に見つかった奴を返す
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
					//// debug
					//if (16721636401499889413LL >> 32 == hash_high) {
					//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
					//	std::cout << "[16721636401499889413][LookUpDirect()] different depth, but return entry because pn == 0." << std::endl;
					//}
#ifdef DEBUG
					if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[LookUpDirect()] unlock" << std::endl;
					}
#endif
#ifdef DEBUG_DEADLOCK_20231014_0
					if (entry.hash_high == HASH_DEBUG_TARGET2 >> 32) {
						std::cout
							<< "[HASH_DEBUG_TARGET2][LookUpDirect()][entry] "
							<< entry.depth << ", " << entry.pn << ", " << entry.dn << std::endl;
					}
#endif
					entry.unlock();
					return entry;
				}
			}
			else if (entry.dn == 0) {
				if (entry.hand.isEqualOrSuperior(hand) && entry.num_searched != REPEAT) {
#ifdef DEBUG
					if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[LookUpDirect()] unlock" << std::endl;
					}
#endif
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

#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
			std::cout << "[LookUpDirect()] unlock" << std::endl;
		}
#endif
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
	uint32_t best_num_searched = UINT_MAX;         // NOTE: num_searched の最小値？
	TTEntry* best_entry_include_mate = nullptr;
	uint32_t best_num_searched_include_mate = UINT_MAX;    // NOTE
	//     : 詰み局面のみのnum_searched 最小値？
	//       名前的には違うっぽいけど。。。

// NOTE
//     : 現状上書きする可能性のあるentry のlock は解放すべきでないはず。
//       一旦解放して、その間に価値のある別の情報が書き込まれたときにそれを潰すことになる。
//       まぁ十分大きな置換表があれば問題ないはず。
//       ただ、30スレッドぐらいで探索するのに、PUCTで64GBぐらいは食うはずで残りの64GBで十分大きいかというのは疑問が残る。
	for (auto& entry : entries.entries) {
#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
			std::cout << "[LookUpDirect()][replacing] lock" << std::endl;
		}
#endif
		entry.lock();
		if (entry.pn != 0) {
			if (best_num_searched > entry.num_searched) {
				if (best_entry != nullptr) {
#ifdef DEBUG
					if (best_entry->hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[LookUpDirect()] unlock" << std::endl;
					}
#endif
					best_entry->unlock();    // 書き換える可能性は無くなったので解放
				}
				best_entry = &entry;
				best_num_searched = entry.num_searched;
			}
			else {
#ifdef DEBUG
				if (best_entry->hash_high == HASH_HIGH_DEBGU_TARGET) {
					std::cout << "[LookUpDirect()] unlock" << std::endl;
				}
#endif
				entry.unlock();
			}
		}
		else {
			if (best_num_searched_include_mate > entry.num_searched) {
				if (best_entry_include_mate != nullptr) {
#ifdef DEBUG
					if (best_entry_include_mate->hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[LookUpDirect()] unlock" << std::endl;
					}
#endif
					best_entry_include_mate->unlock();    // 書き換える可能性は無くなったので解放
				}
				best_entry_include_mate = &entry;
				best_num_searched_include_mate = entry.num_searched;
			}
			else {
#ifdef DEBUG
				if (best_entry_include_mate->hash_high == HASH_HIGH_DEBGU_TARGET) {
					std::cout << "[LookUpDirect()] unlock" << std::endl;
				}
#endif
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

	//// debug
	//if (16721636401499889413LL >> 32 == best_entry->hash_high) {
	//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
	//	std::cout << "[16721636401499889413][LookUpDirect()][replaced] " << ", " << best_entry->pn << ", " << best_entry->dn << ", " << best_entry->depth << ", " << best_entry->hand.value() << std::endl;
	//	exit(1);
	//}

	best_entry->hash_high = hash_high;
	best_entry->hand = hand;
	best_entry->depth = depth;
	best_entry->pn = 1;
	best_entry->dn = 1;
	best_entry->generation = generation;
	best_entry->num_searched = 0;

#ifdef DEBUG
	if (best_entry->hash_high == HASH_HIGH_DEBGU_TARGET) {
		std::cout << "[LookUpDirect()] unlock" << std::endl;
	}
#endif
	best_entry->unlock();
	return *best_entry;
}

template <bool or_node>
TTEntry& TranspositionTable::LookUp(const Position& n) {
	//// debug
	//if (16721636401499889413LL == n.getBoardKey()) {
	//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
	//	std::cout << "[16721636401499889413] " << bts(or_node) << std::endl;
	//}
	
#ifdef DEBUG
	if (HASH_DEBUG_TARGET2 == n.getBoardKey()) {
		std::cout << "[" << static_cast<unsigned long long>(HASH_DEBUG_TARGET2) << "] sfen = " << n.toSFEN() << std::endl;
	}
#endif

	// NOTE: このLookUp<>() はLookUp() のラッパー
	return LookUp(n.getBoardKey(), or_node ? n.hand(n.turn()) : n.hand(oppositeColor(n.turn())), n.gamePly());
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

void TranspositionTable::Reset() {
	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}

	// NOTE: OverFlow とか桁落ちとかを防ぐために、割って掛けて割ってる
	//std::cout << "info string alloc_size=" << (sizeof(Cluster) * (num_clusters / 1024)) / 1024 << "mb" << std::endl;

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
// @arg thpn, thdn
//     : 一番初めのroot ではinf が渡されるっぽい。
//     : この局面に入ってくる直前において、
//       Position n の兄弟局面 の内、2番目に良い局面のpn, dn
//       ※2番目に良い局面とは、
//         OrNode なら2番目にpn が小さい局面、
//         AndNode なら2番目にdn が小さい局面 のことを指す。
template <bool or_node>
void ParallelDfPn::dfpn_inner(
	Position& n, const int thpn, const int thdn/*, bool inc_flag*/, const uint16_t maxDepth, int64_t& searchedNode, const int threadid
) {
	//std::cout << "[dfpn_inner()][start][" << "xxxx" << ", " << "xxxx" << "] done" << std::endl;
	auto& entry = transposition_table.LookUp<or_node>(n);
	//std::cout << "[dfpn_inner()][done][" << entry.num_searched << ", " << entry.depth << "] done" << std::endl;

	entry.lock();

#ifdef HASH_DEBUG_SAFE
	if (THREADID_COND) {
		if (true || entry.hash_high == HASH_DEBUG_TARGET >> 32) {
			std::stringstream ss;
			ss
				<< "[threadid == " << threadid << "][dfpn_inner()][start] ("
				<< entry.pn << ", " << entry.dn << ") "
				<< entry.depth << ", " << entry.num_searched << ", " << entry.hash_high;
			//if (entry.hash_high == HASH_DEBUG_TARGET >> 32) {
			//	ss << " !!HASH_DEBUG_TARGET!!";
			//}
			enqueue_global_debug_str(ss.str());
		}
	}
#endif

	entry.unlock();

	//if (threadid == 0) {
	//	std::cout << "[";
	//}
	//sync_cout << threadid << IO_UNLOCK;
	//std::cout << threadid;

	if (or_node) {
		// NOTE: 深さ制限に達したら、不詰みと判定
		if (n.gamePly() + 1 > maxDepth) {
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
				std::cout << "[dfpn_inner()] lock" << std::endl;
			}
#endif
			entry.lock();
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				entry.num_searched = REPEAT;    // TODO: REPEAT って何？
			}
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
				std::cout << "[dfpn_inner()] unlock" << std::endl;
			}
#endif
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
#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
			std::cout << "[dfpn_inner()] lock" << std::endl;
		}
#endif
		entry.lock();

		if (or_node) {

			// 自分の手番でここに到達した場合は王手の手が無かった、
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
			}
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
#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
			std::cout << "[dfpn_inner()][if(move_picker.empty())] unlock" << std::endl;
		}
#endif
		entry.unlock();
		return;    // NOTE: serach_result = 不詰み || 詰み
	}

	// debug
	//const auto root_key_debug = n.getBoardKey();

	//// NOTE
	////     : 一旦、2 or 3手詰め 探索をする。
	// 新規節点で固定深さの探索を併用

#ifdef DEBUG
	if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
		std::cout << "[dfpn_inner()] lock" << std::endl;
	}
#endif


#ifdef DFPN_MOD_V1
	// TODO: 本当は、証明ゴマのセットとかせんといかん。
	// TODO: lookUpChildEntry で、詰んだ局面もpn=0って設定戦といかん。
	entry.lock();
	if (entry.num_searched == 0) {
		entry.unlock();
		if (or_node) {

			StateInfo si;
			const CheckInfo ci(n);
			for (const auto& ml : move_picker) {    // NOTE: OrNode での指し手を探索
				const Move& m = ml.move;

				n.doMove(m, si, ci, true);

				// 千日手のチェック
				if (n.isDraw(16) == RepetitionWin) {
					// NOTE
					//     : 連続王手の千日手
					//     : if (n.gamePly() + 2 > maxDepth) でgamePly() を使うので、それまではundoMove() 出来ないので個別にundoMove() する。
					// 受け側の反則勝ち
					n.undoMove(m);
					continue;
				}

				auto& entry2 = transposition_table.LookUp<false>(n);

				MovePicker<false> move_picker2(n);

				if (move_picker2.size() == 0) {    // 詰み
					// NOTE
					//     : 後手に逃げる手が無いので、今回の指し手m は詰ませる手である。
					//     : if (n.gamePly() + 2 > maxDepth) でgamePly() を使うので、それまではundoMove() 出来ないので個別にundoMove() する。
					// 1手で詰んだ
					n.undoMove(m);

					// TODO
					//     : kInfinitePnDn + 1 と+1 する理由は何？
					//       -> get_pv_inner とかでも使われてて、使われてるところ見た感じ、
					//          AndNode では+1, OrNode では+2 っぽい。。。?
					//       -> もっと言うと、+1 の時、その局面は詰み局面そのもの？
					//          ほんでもって+2 の時、1手詰め局面かな？

					entry2.lock();
					entry2.pn = 0;
					entry2.dn = kInfinitePnDn + 1;
					entry2.unlock();

					entry.lock();
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

					entry.unlock();
					return;    // NOTE: serach_result = 詰み
				}

				// TODO
				//     : 何故+2 ？
				//     : move_picker より先にコッチチェックすべきでは？？
				//       -> あれか、max move 不詰みで時短するより、詰んだという情報の方が価値が高いって話かな。
				// NOTE: 手数限界を突破したので、不詰み扱い
				if (n.gamePly() + 2 > maxDepth) {
					// NOTE
					//     : if (n.gamePly() + 2 > maxDepth) (すぐ上) でgamePly() を使うので、それまではundoMove() 出来ないので個別にundoMove() する。
					n.undoMove(m);

					entry2.lock();
#ifdef DFPN_MOD_V2
					if (entry2.pn != 0) {
#else
					if (true) {
#endif
						entry2.pn = kInfinitePnDn;
						entry2.dn = 0;
						entry2.num_searched = REPEAT;
					}
					entry2.unlock();

					continue;
				}


				// NOTE
				//     : if (n.gamePly() + 2 > maxDepth) (上) でgamePly() を使うので、それまではundoMove() 出来ないので個別にundoMove() する。
				n.undoMove(m);
			}
		}
#ifdef DFPN_MOD_V1_0
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
					auto& entry1 = transposition_table.LookUp<true>(n);


					entry1.lock();
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

					entry1.unlock();
				}
				else {
					// 詰んでないので、m2で詰みを逃れている。
					// 不詰みチェック
					// 王手がない場合
					MovePicker<true> move_picker2(n);
					if (move_picker2.empty()) {
						auto& entry1 = transposition_table.LookUp<true>(n);
#ifdef DEBUG
						if (entry1.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()] lock" << std::endl;
						}
#endif
						entry1.lock();
#ifdef DFPN_MOD_V2
						if (entry1.pn != 0) {
#else
						if (true) {
#endif
							entry1.pn = kInfinitePnDn;
							entry1.dn = 0;
							// 反証駒
							// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
							entry1.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
						}

						n.undoMove(m2);    // TODO: undoMove() 重いならクリティカルセクションからどかす。
#ifdef DEBUG
						if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()] lock" << std::endl;
						}
#endif
						entry.lock();
#ifdef DFPN_MOD_V2
						if (entry.pn != 0) {
#else
						if (true) {
#endif
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
#ifdef DEBUG
						if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()][if (move_picker2.empty())] unlock" << std::endl;
						}
						if (entry1.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()][if (move_picker2.empty())] unlock" << std::endl;
						}
#endif
						entry.unlock();
						entry1.unlock();
						return;
					}
					n.undoMove(m2);
					goto NO_MATE;
				}

				n.undoMove(m2);
			}

			// すべて詰んだ
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
				std::cout << "[dfpn_inner()] lock" << std::endl;
			}
#endif
			entry.lock();
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			entry.unlock();
			return;

		NO_MATE:;
		}
#endif    // DFPN_MOD_V1_0
	}
	else {
		entry.unlock();
	}
#else

	entry.lock();
	if (entry.num_searched == 0) {
#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
			std::cout << "[dfpn_inner()][if (entry.num_searched == 0)] unlock" << std::endl;
		}
#endif
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
				auto& entry2 = transposition_table.LookUp<false>(n);

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
#ifdef DEBUG
					if (entry2.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[dfpn_inner()][if (move_picker2.size() == 0)] lock" << std::endl;
					}
#endif
					entry2.lock();
					entry2.pn = 0;
					entry2.dn = kInfinitePnDn + 1;
#ifdef DEBUG
					if (entry2.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[dfpn_inner()][if (move_picker2.size() == 0)] unlock" << std::endl;
					}
#endif
					entry2.unlock();
#ifdef DEBUG
					if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[dfpn_inner()][if (move_picker2.size() == 0)] lock" << std::endl;
					}
#endif
					entry.lock();
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

#ifdef DEBUG
					// NOTE: root のOrNode で詰みを見つけたので終了
					if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[dfpn_inner()][if (move_picker2.size() == 0)] unlock" << std::endl;
					}
#endif
					entry.unlock();
					return;    // NOTE: serach_result = 詰み
				}

				// TODO
				//     : 何故+2 ？
				//     : move_picker より先にコッチチェックすべきでは？？
				// NOTE: 手数限界を突破したので、不詰み扱い
				if (n.gamePly() + 2 > maxDepth) {
					n.undoMove(m);

#ifdef DEBUG
					if (entry2.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[dfpn_inner()] lock" << std::endl;
					}
#endif
					entry2.lock();
#ifdef DFPN_MOD_V2
					if (entry2.pn != 0) {
#else
					if (true) {
#endif
						entry2.pn = kInfinitePnDn;
						entry2.dn = 0;
						entry2.num_searched = REPEAT;
					}
#ifdef DEBUG
					if (entry2.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[dfpn_inner()][if (n.gamePly() + 2 > maxDepth)] unlock" << std::endl;
					}
#endif
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

					if (n.mateMoveIn1Ply<mmin1ply_Additional>()) {
						auto& entry1 = transposition_table.LookUp<true>(n);

#ifdef DEBUG
						if (entry1.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()] lock" << std::endl;
						}
#endif
						entry1.lock();
						entry1.pn = 0;
						entry1.dn = kInfinitePnDn + 2;
#ifdef DEBUG
						if (entry1.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()][if (n.mateMoveIn1Ply<mmin1ply_Additional>())] unlock" << std::endl;
						}
#endif
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
#ifdef DEBUG
				if (entry2.hash_high == HASH_HIGH_DEBGU_TARGET) {
					std::cout << "[dfpn_inner()] lock" << std::endl;
				}
#endif
				entry2.lock();
				entry2.pn = 0;
				entry2.dn = kInfinitePnDn;
				entry2.unlock();
#ifdef DEBUG
				if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
					std::cout << "[dfpn_inner()] lock" << std::endl;
				}
#endif
				entry.lock();
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
				entry.unlock();

				return;

				// NOTE
				//     : 今回探索したOrNode の指し手は、後手になんらかの逃げ道があった
			NEXT_CHECK:;
				n.undoMove(m);
#ifdef DEBUG
				if (entry2.hash_high == HASH_HIGH_DEBGU_TARGET) {
					std::cout << "[dfpn_inner()][NEXT_CHECK] lock" << std::endl;
				}
#endif
				entry2.lock();
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
#ifdef DEBUG
				if (entry2.hash_high == HASH_HIGH_DEBGU_TARGET) {
					std::cout << "[dfpn_inner()][NEXT_CHECK] unlock" << std::endl;
				}
#endif
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
					auto& entry1 = transposition_table.LookUp<true>(n);

#ifdef DEBUG
					if (entry1.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[dfpn_inner()] lock" << std::endl;
					}
#endif
					entry1.lock();
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
#ifdef DEBUG
					if (entry1.hash_high == HASH_HIGH_DEBGU_TARGET) {
						std::cout << "[dfpn_inner()][if (const Move move = n.mateMoveIn1Ply<mmin1ply_Additional>())] unlock" << std::endl;
					}
#endif
					entry1.unlock();
				}
				else {
					// 詰んでないので、m2で詰みを逃れている。
					// 不詰みチェック
					// 王手がない場合
					MovePicker<true> move_picker2(n);
					if (move_picker2.empty()) {
						auto& entry1 = transposition_table.LookUp<true>(n);
#ifdef DEBUG
						if (entry1.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()] lock" << std::endl;
						}
#endif
						entry1.lock();
#ifdef DFPN_MOD_V2
						if (entry1.pn != 0) {
#else
						if (true) {
#endif
							entry1.pn = kInfinitePnDn;
							entry1.dn = 0;
							// 反証駒
							// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
							entry1.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
						}

						n.undoMove(m2);    // TODO: undoMove() 重いならクリティカルセクションからどかす。
#ifdef DEBUG
						if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()] lock" << std::endl;
						}
#endif
						entry.lock();
#ifdef DFPN_MOD_V2
						if (entry.pn != 0) {
#else
						if (true) {
#endif
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
#ifdef DEBUG
						if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()][if (move_picker2.empty())] unlock" << std::endl;
						}
						if (entry1.hash_high == HASH_HIGH_DEBGU_TARGET) {
							std::cout << "[dfpn_inner()][if (move_picker2.empty())] unlock" << std::endl;
						}
#endif
						entry.unlock();
						entry1.unlock();
						return;
					}
					n.undoMove(m2);
					goto NO_MATE;
				}

				n.undoMove(m2);
			}

			// すべて詰んだ
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
				std::cout << "[dfpn_inner()] lock" << std::endl;
			}
#endif
			entry.lock();
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			entry.unlock();
			return;

		NO_MATE:;
		}

	}
	else {
#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET) {
			std::cout << "[dfpn_inner()][before isDraw()] unlock" << std::endl;
		}
#endif
		entry.unlock();
	}
#endif    // DFPN_MOD_V1

	// TODO
	//     : どうしてこのタイミングで千日手判定するんや？
	//       -> 恐らくやけど、dfpn 呼ぶ局面では詰みの方が珍しいからやないかな。
	//          (先に短手数の詰みを確認しておいた方が、この関数を抜けると期待される時間は小さくなる。)
	//     : 引き分けなら評価値を書き換えるやつ、
	//       例えばdepth=6とdepth=9 で同じ局面が出てきた場合、depth=6 の詰み不詰みも千日手として値が書き換えられるが、
	//       qhapaq の記事のように、千日手は入り口から一番遠い局面で初めて不詰み扱いしなければならないので、
	//       depth=6 では不詰みとしてはいけない(はず)。「
	// 千日手のチェック
	switch (n.isDraw(16)) {
	case RepetitionWin:
		//cout << "RepetitionWin" << endl;
		// 連続王手の千日手による勝ち
		if (or_node) {
			// ここは通らないはず
			//if (entry.dn == 0) {
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] lock" << std::endl;
			}
#endif
			entry.lock();
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			entry.num_searched = REPEAT;
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] unlock" << std::endl;
			}
#endif
			entry.unlock();
			//}
		}
		else {
			//if (entry.pn != 0) {
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] lock" << std::endl;
			}
#endif
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
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] unlock" << std::endl;
			}
#endif
			entry.unlock();
			//}
		}
		return;

	case RepetitionLose:
		//cout << "RepetitionLose" << endl;
		// 連続王手の千日手による負け
		if (or_node) {
			//std::cout << "[RepetitionLose][" << entry.num_searched << ", " << entry.depth << "]" << std::endl;
			//if (entry.pn != 0) {
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] lock" << std::endl;
			}
#endif
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
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] unlock" << std::endl;
			}
#endif
			entry.unlock();
			//}
		}
		else {
			// ここは通らないはず
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] lock" << std::endl;
			}
#endif
			entry.lock();
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			entry.num_searched = REPEAT;
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] unlock" << std::endl;
			}
#endif
			entry.unlock();
		}
		return;

	case RepetitionDraw:
		//std::cout << "[RepetitionDraw][" << entry.num_searched << ", " << entry.depth << "]" << std::endl;
		//cout << "RepetitionDraw" << endl;
		// 普通の千日手
		// ここは通らないはず
#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
			std::cout << "[dfpn_inner()][Draw] lock" << std::endl;
		}
#endif
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
#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
			std::cout << "[dfpn_inner()][Draw] unlock" << std::endl;
		}
#endif
		entry.unlock();
		return;

	case RepetitionSuperior:
		if (!or_node) {
			// NOTE
			//     : AndNode で優越局面になった場合、AndNode 視点で有利になってる訳で、
			//       OrNode 視点ではこの局面に遷移すべきでない。
			//       従って、OrNode 視点で"マイナスの評価値" となるようにpn = inf とする。
			// ANDノードで優越局面になっている場合、除外できる(ORノードで選択されなくなる)

			//if (entry.pn != 0) {
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] lock" << std::endl;
			}
#endif
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
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][Draw] unlock" << std::endl;
			}
#endif
			entry.unlock();
			//}
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

	//if (threadid == 0) {
	//	std::cout << "]";
	//}

	// NOTE
	//     : 任意のnode の探索数はmaxSearchNode を超えてはならない。
	while (searchedNode < maxSearchNode && !stop) {

		// NOTE: 恐らくnum_searched について代入でない更新がなされるのはここだけ。
#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
			std::cout << "[dfpn_inner()][while, num_searched] lock" << std::endl;
		}
#endif
		//std::cout << "[inc]";
		entry.lock();
		++entry.num_searched;
		//if (threadid == 0) {
		//	//std::cout << "/" << entry.depth << "," << entry.num_searched << "]";
		//	std::cout << "/" << entry.depth << "]";
		//}


#ifdef HASH_DEBUG_SAFE
		if (THREADID_COND) {
			if (true || entry.hash_high == HASH_DEBUG_TARGET >> 32) {
				std::stringstream ss;
				ss
					<< "[threadid == " << threadid << "][dfpn_inner()][loop] ("
					<< entry.pn << ", " << entry.dn << ") "
					<< entry.depth << ", " << entry.num_searched << ", " << entry.hash_high;
				//if (entry.hash_high == HASH_DEBUG_TARGET >> 32) {
				//	ss << " !!HASH_DEBUG_TARGET!!";
				//}
				enqueue_global_debug_str(ss.str());
			}
		}
#endif

#ifdef DEBUG
		if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
			std::cout << "[dfpn_inner()][while, num_searched] unlock" << std::endl;
		}
#endif
		entry.unlock();

		// debug
		//if (true) {
		//	std::cout << "[dfpn_inner()]" << entry.num_searched << ", " << entry.pn << ", " << entry.dn << std::endl;
		//}

		//// debug
		//if (16721636401499889413LL >> 32 == entry.hash_high) {
		//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
		//	std::cout << "[16721636401499889413][dfpn_inner()][start]" << entry.num_searched << ", " << entry.pn << ", " << entry.dn << std::endl;
		//}

		//// debug
		//// こっちは毎度実行される。
		//Hand tmpHand(4196361);
		//auto&& tmpEntry = transposition_table.LookUp(16721636401499889413, tmpHand, 6);
		//if (16721636401499889413LL >> 32 == tmpEntry.hash_high) {
		//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
		//	std::cout << "[16721636401499889413][dfpn_inner()][" << entry.num_searched << ", " << entry.depth << "] (" << tmpEntry.pn << ", " << tmpEntry.dn << ") " << tmpEntry.depth << ", " << tmpEntry.hand.value() << std::endl;
		//}

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
				//const auto& child_entry = transposition_table.LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				auto& child_entry = transposition_table.LookUpDirect(
					*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1
				);    // 本当はconst auto&

#ifdef DEBUG
				if (child_entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
					std::cout << "[dfpn_inner()][while, or_node, for, child_entry] lock" << std::endl;
				}
#endif
				child_entry.lock();

				if (child_entry.pn == 0) {    // NOTE: OrNode で子ノードの一つが詰みだったので、現在のノードも詰み
					// 詰みの場合
					//cout << n.toSFEN() << " or" << endl;
					//cout << bitset<32>(entry.hand.value()) << endl;
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
#ifdef DEBUG
						if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
							std::cout << "[dfpn_inner()][while, or_node, for (const auto& move : move_picker), if (child_entry.pn == 0), if] lock" << std::endl;
						}
#endif
						entry.lock();
						if (entry.hand.numOf(hp) > child_entry.hand.numOf(hp)) {
							entry.hand = child_entry.hand;
							entry.hand.plusOne(move.move.handPieceDropped());    // TODO: 何故、これ、.plusOne(hp); にしない？
						}
#ifdef DEBUG
						if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
							std::cout << "[dfpn_inner()][while, or_node, for (const auto& move : move_picker), if (child_entry.pn == 0), if] unlock" << std::endl;
						}
#endif
						entry.unlock();
					}
					// TODO
					//     : 駒を取る場合は、問答無用で今回の指し手についての証明駒を採用するっぽい。何故？？
					// 後手の駒を取る手ならば、証明駒から削除する
					else {
						const Piece to_pc = n.piece(move.move.to());
						if (to_pc != Empty) {    // NOTE: 駒を取る手である
#ifdef DEBUG
							if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
								std::cout << "[dfpn_inner()][while, or_node, for (const auto& move : move_picker), if (child_entry.pn == 0), else] lock" << std::endl;
							}
#endif
							entry.lock();
							entry.hand = child_entry.hand;
							const PieceType pt = pieceToPieceType(to_pc);
							const HandPiece hp = pieceTypeToHandPiece(pt);
							if (entry.hand.exists(hp)) {
								entry.hand.minusOne(hp);
							}
#ifdef DEBUG
							if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
								std::cout << "[dfpn_inner()][while, or_node, for (const auto& move : move_picker), if (child_entry.pn == 0), else] unlock" << std::endl;
							}
#endif
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
				// NOTE
				//     : root node(entry) が不詰みなら基本ここを通る(はず)。
				//       (置換表から貰った局面(entry)では不詰みと言い切れたけどこの局面では違う場合(子ノードで詰んでる局面アリ)、上のif にい)
				//     : 以下のように、反証駒が登録されてる任意の子ノードの、任意の反証駒(ex 歩, 香, 桂馬, ...) についての最小値を求めれば、それが則ち積集合である。
#ifdef DFPN_MOD_V0
				else if(true) {
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
#ifdef DEBUG
				if (child_entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
					std::cout << "[dfpn_inner()][while, or_node, for, child_entry] unlock" << std::endl;
				}
#endif
				child_entry.unlock();
				//++tmpCount;    // debug
			}    // NOTE: for 終了

#ifdef DEBUG
#ifdef DFPN_MOD_V0
			//std::cout << "debug: entry = (" << entry.pn << ", " << entry.dn << "), " << entry_pn << ", " << entry_dn << std::endl;
#else
			std::cout << "debug: entry = (" << entry.pn << ", " << entry.dn << ")" << std::endl;
#endif
#endif

			// TODO
			//     : kInfinitePnDn がどういう場面で代入されるか 把握できてない。
			//     : これは、子ノードに複数dn = kInfinitePnDn な奴が合った場合にclip するのが狙い？
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][while, or_node, entry] lock" << std::endl;
			}
#endif
			entry.lock();
#ifdef DFPN_MOD_V0
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
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
				if constexpr (EPS == 0) {
					thpn_child = std::min(thpn, second_best_pn + 1);
				}
				else if constexpr (EPS > 0) {
					// より子ノードの長く滞在
					thpn_child = std::min(thpn, static_cast<int>(_EPS_PLUS_ONE * second_best_pn + 1));
				}
				else {
					// より子ノードに短く滞在
					thpn_child = std::min(
						thpn,
						best_pn + static_cast<int>(_EPS_PLUS_ONE * (second_best_pn - best_pn) + 1)
					);
				}
				thdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn);
			}
			// TODO
			//     : 一応、if (entry.dn == 0) の場合は最後にunlock() して、
			//       else の場合はthdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn); 
			//       だけを先にやってunlock() すればちょっと速くなるはず。。。
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][while, or_node, entry] unlock" << std::endl;
			}
#endif
			entry.unlock();
		}
		else {
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
				//const auto& child_entry = transposition_table.LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				auto& child_entry = transposition_table.LookUpDirect(
					*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1
				);    // 本当はconst auto&
#ifdef DEBUG
				if (child_entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
					std::cout << "[dfpn_inner()][while, and_node, for, child_entry] lock" << std::endl;
				}
#endif
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
#ifdef DEBUG
					if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
						std::cout << "[dfpn_inner()][while, and_node, for (const auto& move : move_picker), if (child_entry.dn == 0), if] lock" << std::endl;
					}
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
#ifdef DEBUG
					if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
						std::cout << "[dfpn_inner()][while, and_node, for (const auto& move : move_picker), if (child_entry.dn == 0), if] unlock" << std::endl;
					}
#endif
					entry.unlock();
#ifdef DEBUG
					if (child_entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
						std::cout << "[dfpn_inner()][while, and_node, for, child_entry] unlock" << std::endl;
					}
#endif
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
					best_move = move;
				}
				else if (child_entry.dn < second_best_dn) {
					second_best_dn = child_entry.dn;
				}
#ifdef DEBUG
				if (child_entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
					std::cout << "[dfpn_inner()][while, and_node, for, child_entry] unlock" << std::endl;
				}
#endif
				child_entry.unlock();
			}    // NOTE: for 終了
#ifdef DEBUG
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][while, and_node, entry] lock" << std::endl;
			}
#endif
			entry.lock();
#ifdef DFPN_MOD_V0
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				entry.pn = std::min(entry_pn, kInfinitePnDn);
				entry.dn = entry_dn;
			}
#else
			entry.pn = std::min(entry.pn, kInfinitePnDn);
#endif

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
			if (entry.hash_high == HASH_HIGH_DEBGU_TARGET2) {
				std::cout << "[dfpn_inner()][while, and_node, entry] lock" << std::endl;
			}
			entry.unlock();
		}

		//// debug
		//if (16721636401499889413LL >> 32 == entry.hash_high) {
		//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
		//	std::cout << "[16721636401499889413][dfpn_inner()][done]" << entry.num_searched << ", " << entry.pn << ", " << entry.dn << std::endl;
		//}

		// if (pn(n) >= thpn || dn(n) >= thdn)
		//   break; // termination condition is satisfied
		if (entry.pn >= thpn || entry.dn >= thdn) {
#ifdef HASH_DEBUG_SAFE
			if (THREADID_COND) {
				if (true || entry.hash_high == HASH_DEBUG_TARGET >> 32) {
					std::stringstream ss;
					ss
						<< "[threadid == " << threadid << "][dfpn_inner()][end search] ("
						<< entry.pn << ", " << entry.dn << ") "
						<< entry.depth << ", " << entry.num_searched << ", " << entry.hash_high;
					//if (entry.hash_high == HASH_DEBUG_TARGET >> 32) {
					//	ss << " !!HASH_DEBUG_TARGET!!";
					//}
					enqueue_global_debug_str(ss.str());
				}
			}
#endif
			break;
		}

		StateInfo state_info;
		//cout << n.toSFEN() << "," << best_move.toUSI() << endl;
		n.doMove(best_move, state_info);
		++searchedNode;
		//if (threadid == 0) {
		//	std::cout << "n";
		//}
		dfpn_inner<!or_node>(n, thpn_child, thdn_child/*, inc_flag*/, maxDepth, searchedNode, threadid);
		n.undoMove(best_move);

	}
	//if (threadid == 0) {
	//	entry.lock();
	//	//std::cout << "<" << entry.pn << "," << entry.dn << "," << entry.depth << ">";
	//	std::cout << "<" << entry.depth << ">";
	//	entry.unlock();
	//}
}

// 詰みの手返す
Move ParallelDfPn::dfpn_move(Position& pos) {
	MovePicker<true> move_picker(pos);
	for (const auto& move : move_picker) {
		const auto& child_entry = transposition_table.LookUpChildEntry<true>(pos, move);
		if (child_entry.pn == 0) {
			return move;
		}
	}

	return Move::moveNone();
}

template<bool or_node, bool safe>
int ParallelDfPn::get_pv_inner(Position& pos, std::vector<Move>& pv) {
	std::stringstream ss;
	pos.print(ss);
	if (or_node) {
		//std::cout << "[Or][ply = " << pos.gamePly() << "][" << pos.getBoardKey() << "]\n" << ss.str() << std::endl;
		// ORノードで詰みが見つかったらその手を選ぶ
		MovePicker<true> move_picker(pos);
		for (const auto& move : move_picker) {
			// NOTE: コイツはlock, unlock 以外はconst
			auto& child_entry = transposition_table.LookUpChildEntry<true>(pos, move);
			if (safe) child_entry.lock();

			//std::cout
			//	<< "[Or][ply = " << pos.gamePly() << "][move_picker] " << Move(move).toUSI()
			//	<< " = [" << child_entry.pn << ", " << child_entry.dn << ", " << child_entry.hand.value() << "]" << std::endl;
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
		//std::cout << "[Or][ply = " << pos.gamePly() << "] loop end." << std::endl;
		//// debug
		//Hand tmpHand(4196361);
		//auto&& tmpEntry = transposition_table.LookUp(16721636401499889413, tmpHand, 6);
		//if (16721636401499889413LL >> 32 == tmpEntry.hash_high) {
		//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
		//	std::cout << "[16721636401499889413][get_pv_inner()] " << tmpEntry.num_searched << ", " << tmpEntry.pn << ", " << tmpEntry.dn << ", " << tmpEntry.hand.value() << std::endl;
		//}
	}
	else {
		//std::cout << "[And][ply = " << pos.gamePly() << "][" << pos.getBoardKey() << "]\n" << ss.str() << std::endl;
		// ANDノードでは詰みまでが最大手数となる手を選ぶ
		int max_depth = 0;
		std::vector<Move> max_pv;    // NOTE: 最大のPV
		MovePicker<false> move_picker(pos);
		for (const auto& move : move_picker) {
			// NOTE: コイツはlock, unlock 以外はconst
			auto& child_entry = transposition_table.LookUpChildEntry<false>(pos, move);
			if (safe) child_entry.lock();

			if (child_entry.pn == 0) {
				std::vector<Move> tmp_pv{ move };
				StateInfo state_info;
				pos.doMove(move, state_info);
				int depth = -kInfinitePnDn;
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
std::tuple<std::string, int, Move> ParallelDfPn::get_pv(Position& pos) {
	// debug
	//Hand tmpHand(4196361);
	//auto&& tmpEntry = transposition_table.LookUp(16721636401499889413, tmpHand, 6);
	//if (16721636401499889413LL >> 32 == tmpEntry.hash_high) {
	//	// hand = 0b10000000000100000001001, つまりpawn 9, Knight 2, bishop 1 なので恐らく例の局面
	//	std::cout << "[16721636401499889413][get_pv()][] " << tmpEntry.num_searched << ", " << tmpEntry.pn << ", " << tmpEntry.dn << ", " << tmpEntry.hand.value() << std::endl;
	//}

	std::vector<Move> pv;
	int depth = -1;
	if constexpr (safe) {
		depth = get_pv_inner<true, true>(pos, pv);
	}
	else {    // unsafe
		depth = get_pv_inner<true, false>(pos, pv);
	}
	if (pv.size() == 0) {
		pv[0] = Move(0);
	}
	const Move& best_move = pv[0];
	std::stringstream ss;
#ifdef DEBUG
	std::cout << "info: get_pv(), pv.size() = " << pv.size() << std::endl;
	std::cout << "info: get_pv(), best_move.toUSI() = " << best_move.toUSI() << std::endl;
#endif
	ss << best_move.toUSI();
	for (size_t i = 1; i < pv.size(); i++)
		ss << " " << pv[i].toUSI();

	return std::make_tuple(ss.str(), depth, best_move);
}

void ParallelDfPn::init()
{
	transposition_table.Resize(HASH_SIZE_MB);
}

// 詰将棋探索のエントリポイント
bool ParallelDfPn::dfpn(Position& r, int64_t& searched_node, const int threadid) {
	// キャッシュの世代を進める

	//std::cout << "debug: DLSHOGI dfpn::dfpn" << std::endl;

	searched_node = 0;    // NOTE: 探索ノード数をreset
	if (!r.inCheck()) {
		// 1手詰みチェック
		Move mate1ply = r.mateMoveIn1Ply<mmin1ply_Additional>();
		if (mate1ply) {
			auto& child_entry = transposition_table.LookUpChildEntry<true>(r, mate1ply);
			child_entry.lock();
			child_entry.pn = 0;
			child_entry.dn = kInfinitePnDn + 1;
			child_entry.unlock();
			return true;
		}
	}
	dfpn_inner<true>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, std::min(r.gamePly() + kMaxDepth, draw_ply), searched_node, threadid);
	auto&& entry = transposition_table.LookUp<true>(r);

	//cout << searched_node << endl;

	/*std::vector<Move> moves;
	std::unordered_set<Key> visited;
	dfs(true, r, moves, visited);
	for (Move& move : moves)
	cout << move.toUSI() << " ";
	cout << endl;*/

	entry.lock();
	const auto&& retVal = (entry.pn == 0);
	sync_cout << "info: dfpn done, threadid = " << threadid << ", retVal = " << bts(retVal) << std::endl;
	std::cout
		<< "[key = " << r.getBoardKey()
		<< "][high = " << (r.getBoardKey() >> 32) << "]" << std::endl;
	this->_print_entry_info(entry);
	entry.unlock();
	std::string pv_str;
	int depth;
	Move bestmove;
	std::tie(pv_str, depth, bestmove) = this->get_pv<true>(r);
	std::cout << "info: depth = [" << depth << "]" << std::endl;
	std::cout << "info: bestmove = [" << bestmove.toUSI() << "]" << std::endl;
	std::cout << "info: pv_str = [" << pv_str << "]" << std::endl;
	//flush_global_debug_str();
	std::cout << IO_UNLOCK;


	return retVal;
}

// 詰将棋探索のエントリポイント
bool ParallelDfPn::dfpn_andnode(Position& r, int64_t& searched_node, const int threadid) {
	// 自玉に王手がかかっていること

	// キャッシュの世代を進める
	//transposition_table.NewSearch();

	searched_node = 0;
	dfpn_inner<false>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, std::min(r.gamePly() + kMaxDepth, draw_ply), searched_node, -1 /* dummy threadid */);
	auto& entry = transposition_table.LookUp<false>(r);

	entry.lock();
	const auto&& retVal = (entry.pn == 0);
	entry.unlock();

	return retVal;
}

template<bool or_node>
void ParallelDfPn::print_entry_info(Position& n) {
	TTEntry& entry = transposition_table.LookUp<or_node>(n);
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
template std::tuple<std::string, int, Move> ParallelDfPn::get_pv<true>(Position& pos);
template std::tuple<std::string, int, Move> ParallelDfPn::get_pv<false>(Position& pos);

template void ParallelDfPn::print_entry_info<true>(Position& n);
template void ParallelDfPn::print_entry_info<false>(Position& n);

#endif    // DFPN_PARALLEL3