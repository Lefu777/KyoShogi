#include <unordered_set>

#include "my_dfpn_parallel5.hpp"

#include "debug_string_queue.hpp"

#include <cmath>

#ifdef DFPN_PARALLEL5

//constexpr int skipSize[]  = { 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
//constexpr int skipPhase[] = { 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

constexpr int skipSize[] = { 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
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

//constexpr uint64_t DEBUG_HASH_0 = 12033080882891618798ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_0 = DEBUG_HASH_0 >> 32;
//constexpr uint64_t DEBUG_HASH_P_0 = 2027170531019441307ULL;
//constexpr uint64_t DEBUG_HASH_P_HIGH_0 = DEBUG_HASH_P_0 >> 32;
//
//constexpr uint64_t DEBUG_HASH_1 = 15344768792386519950ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_1 = DEBUG_HASH_1 >> 32;
//constexpr uint64_t DEBUG_HASH_P_1 = 9579420516929808861ULL;
//constexpr uint64_t DEBUG_HASH_P_HIGH_1 = DEBUG_HASH_P_1 >> 32;
//
//// skip 目的only
//constexpr uint64_t DEBUG_HASH_2_SKIP = 9474161064609960782ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_2_SKIP = DEBUG_HASH_2_SKIP >> 32;
//constexpr uint64_t DEBUG_HASH_P_2_SKIP = 7098768730445593323ULL;
//constexpr uint64_t DEBUG_HASH_P_HIGH_2_SKIP = DEBUG_HASH_P_2_SKIP >> 32;
//
//
//constexpr uint64_t DEBUG_HASH_3 = 17035644172859462960ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_3 = DEBUG_HASH_3 >> 32;
//constexpr uint64_t DEBUG_HASH_P_3 = 3917466325995148055ULL;
//constexpr uint64_t DEBUG_HASH_P_HIGH_3 = DEBUG_HASH_P_3 >> 32;
//
//constexpr uint64_t DEBUG_HASH_4 = 13795971860294232562ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_4 = DEBUG_HASH_4 >> 32;
//constexpr uint64_t DEBUG_HASH_P_4 = 901289320336155321ULL;
//constexpr uint64_t DEBUG_HASH_P_HIGH_4 = DEBUG_HASH_P_4 >> 32;
//
//constexpr uint64_t DEBUG_HASH_P_5 = 12456332060767114348ULL;
//constexpr uint64_t DEBUG_HASH_P_HIGH_5 = DEBUG_HASH_P_5 >> 32;
//
//
//constexpr uint64_t DEBUG_HASH_6 = 12456332060767114348ULL;
//constexpr uint64_t DEBUG_HASH_HIGH_6 = DEBUG_HASH_6 >> 32;
//constexpr uint64_t DEBUG_HASH_P_6 = 7127765833921877639ULL;
//constexpr uint64_t DEBUG_HASH_P_HIGH_6 = DEBUG_HASH_P_6 >> 32;
//
//constexpr uint64_t DEBUG_HASH_HIGH_TAG = 2900215811ULL;

//constexpr uint64_t DEBUG_HASH_0 = 17368764834879979320ULL;
//constexpr uint64_t DEBUG_HASH_0 = 9904918893967150897ULL;

// pn = inf とset される瞬間にフォーカス
//constexpr uint64_t DEBUG_HASH_0 = 9904918893967150897ULL;
//constexpr uint64_t DEBUG_HASH_0 = 2298797676493542762ULL;
//constexpr uint64_t DEBUG_HASH_0 = 4924085297936155658ULL;
//constexpr uint64_t DEBUG_HASH_0 = 14626282299106150835ULL;
//constexpr uint64_t DEBUG_HASH_0 = 8014045521880561364ULL;
//constexpr uint64_t DEBUG_HASH_0 = 12646899016840140591ULL;
//constexpr uint64_t DEBUG_HASH_0 = 2880970618142916390ULL;
//constexpr uint64_t DEBUG_HASH_0 = 8778941955408426147ULL;
//constexpr uint64_t DEBUG_HASH_0 = 2507235211485217628ULL;
//constexpr uint64_t DEBUG_HASH_0 = 13490133344281940821ULL;
//constexpr uint64_t DEBUG_HASH_0 = 25030004637123026ULL;
//constexpr uint64_t DEBUG_HASH_0 = 10487513834830542361ULL;
//constexpr uint64_t DEBUG_HASH_0 = 5509345658865768792ULL;
//constexpr uint64_t DEBUG_HASH_0 = 4822201167466198853ULL;
//constexpr uint64_t DEBUG_HASH_0 = 5087548461681687616ULL;
//constexpr uint64_t DEBUG_HASH_0 = 4016205446513378681ULL;
//constexpr uint64_t DEBUG_HASH_0 = 14077091781301672854ULL;
//constexpr uint64_t DEBUG_HASH_0 = 15550032291875106951ULL;
constexpr uint64_t DEBUG_HASH_0 = 0;

//constexpr uint64_t DEBUG_CH_HASH_0 = 15550032291875106951ULL;
constexpr uint64_t DEBUG_CH_HASH_0 = 0;

// hash が一致したら局面を表示する。
//constexpr uint64_t DEBUG_HASH_PRINT_0 = 6260150082945499370ULL;
constexpr uint64_t DEBUG_HASH_PRINT_0 = 0;


// NOTE
//    : depth = 36 ではok, depth=25 ならアウト
//      depth=27 もいけてそう。。。

//constexpr int DEBUG_HASH_0_DEPTH = 1;

// get_pv_inner で情報を表示したい局面。
//constexpr uint64_t DEBUG_HASH_1 = 2298797676493542762ULL;
//constexpr uint64_t DEBUG_HASH_1 = 6584885400792202254ULL;
//constexpr uint64_t DEBUG_HASH_1 = 4768778969317999469ULL;
//constexpr uint64_t DEBUG_HASH_1 = 15136776804311804116ULL;
//constexpr uint64_t DEBUG_HASH_1 = 4924085297936155658ULL;
constexpr uint64_t DEBUG_HASH_1 = 0;


//constexpr uint64_t DEBUG_CHILD_HASH_0 = 5949773176545622198ULL;
//constexpr uint64_t DEBUG_CHILD_HASH_0 = 14626282299106150835ULL;


// debug 時に出力する条件
//#define THREADID_COND (threadid == 1)


//// Impl
using namespace std;
using namespace ns_dfpn;

// MateMoveIn1Ply<>() のadditional
constexpr bool mmin1ply_Additional = true;
// cluster_mask の上限。
// cluster にアクセスするkey として使えるのは64bit hash の内下位32bit まで。
// (上位32bit はentry に格納している。cluster にアクセスするkey が上位32bit に例えばN bit侵食してしまうと、
//   衝突する確率は1/(2**32) から1/(2**(32-N)) に上がってしまう。)。
constexpr uint64_t CLUSTERS_MASK_LIMIT = 4294967296ULL - 1;    // 2**32 - 1

////////// test(@20231104)
//// 1 + epx
//// https://link.springer.com/chapter/10.1007/978-3-319-09165-5_12
//constexpr float EPS = 0;
//constexpr float _EPS_PLUS_ONE = 1 + EPS;
//
//// 1 も 3 も 4 も速度が落ちた
//// TH_MULTIPLY_LOG_BASE の時はTHPN_MULTIPLY を統一して使う。
//constexpr float THPN_MULTIPLY = 1;
//constexpr float THDN_MULTIPLY = 1;
//// 現在のroot のentry のnum_searched がTHPN_MULTIPLY 未満の時にのみTHPN_MULTIPLY を適用する。
//// (選択した子ノードのnum_searched でやるべきだと思う。)
//// num_searched は、スレッド数に応じて増えていくので、そのあたりの調整は必要。
//constexpr uint32_t TH_MULTIPLY_NS_THRESHOLD = 1000;
//
//// TODO: 2**n を取れば、lzcnt 使って近似できるっちゃ出来る。
//constexpr int TH_MULTIPLY_LOG_BASE = 16;

//////// 現状の最適(@20231103)
// 1 + epx
// https://link.springer.com/chapter/10.1007/978-3-319-09165-5_12
constexpr float EPS = 1;
constexpr float _EPS_PLUS_ONE = 1 + EPS;

constexpr uint32_t EPS_MULTIPLY_NS_THRESHOLD = 1000;
constexpr int EPS_MULTIPLY_LOG_BASE = 16;


// thpn_child, thdn_child に掛ける値
constexpr float TH_CH_MULTIPLY = 2;

// 現在のroot のentry のnum_searched がTHPN_MULTIPLY 未満の時にのみTHPN_MULTIPLY を適用する。
// (選択した子ノードのnum_searched でやるべきだと思う。)
// num_searched は、スレッド数に応じて増えていくので、そのあたりの調整は必要。
// -> 1thread での最適値 * n_threads で良い感じかも？
constexpr uint32_t TH_CH_MULTIPLY_NS_THRESHOLD = 1000;

// TODO: 2**n となるように取れば、lzcnt 使って近似できるっちゃ出来る。
constexpr int TH_MULTIPLY_LOG_BASE = 16;


// TODO
//     : eps もlog スケールに。
//     : num_searched は、best_child のを使うようにした方が良い気もする。
//     ; eps, th_ch_multiply をスレッドごとに変えてやればスレッドごとに挙動が変わって局面分散するのでは？
//     : pv をn 手与えてやれば、スレッド数に応じてスケールしたのは、
//       結局は、一番深い所をroot として探索したスレッドのお陰説。

template<typename T>
inline bool max_clip(T& x, const T& max_value) {
	if (x > max_value) {
		x = max_value;
		return true;
	}
	return false;
}

template<typename T>
inline bool min_clip(T& x, const T& min_value) {
	if (x < min_value) {
		x = min_value;
		return true;
	}
	return false;
}

template<typename T>
inline bool min_max_clip(T& x, const T& min_value, const T& max_value) {
	if (min_clip<T>(x, min_value)) {
		return true;
	}
	if (max_clip<T>(x, max_value)) {
		return true;
	}
	return false;
}

template<typename T>
inline int logbase(T a, T base) {
	T retVal = static_cast<T>(log(a) / log(base));
	min_clip<T>(retVal, 1);
	return retVal;
}


// NOTE
//     : 現状、pn_inf はinf しかset されないので、
//       if (should_update_hand()) の中ではわざわざ代入しない。(変わることが無いので。)]
// NOTE
//     : should_update_hand() == false だったとしても、pn = 0, dn = inf が違法となることはない。
//           : 元々がpn !=0 だった場合は、このentry は現局面とhash, depth, hand が全て等しい。
//             この場合は完全に同じ局面とみなす。
//             (異なる局面のentryが返っている可能性があるのは、pn == 0 or dn == 0の時のみ)
//           : 元々pn == 0 だった場合は、なんの変化も無いので問題ない。
//             ただ、1手詰めとしてdn = inf + 2 を登録したのに、hand のupdate をしなかった場合が微妙。
//              -> inf + 2等の値は、should_update_hand() == true の時のみ行うようにすべし。
//     : pn != 0 && dn != 0 のentry は、汎用性が全く無い(他の局面で用いる事が出来ない情報)ので、
//       今回の更新によって必ず汎用性が向上する。それゆえ問答無用で更新して良い。
//     : 詰みの時、pn の値は一意に定まり pn == 0 だが、dn はinf, inf + 1, inf + 2 等種類がある。
//       そのため、今の情報でhand を更新したなら、dn も現状に合わせた値にする。
// 最大手数引き分け不詰で、dn = 0 とset する時は呼び出してはダメ!
#define UpdateDnZero(entry, pn_inf, new_hand_moveable)\
do {\
    if ((entry).dn != 0) {\
        /* debug */\
        if ((entry).hash_high == DEBUG_HASH_0 >> 32) {\
            sync_cout << "debug: pn=inf at UpdateDnZero()" << ",id=" << threadid << ",pn=" << (entry).pn << ",dn=" << (entry).dn << ",depth=" << (entry).depth << ",hand=" << (entry).hand.value() << ",num_searched=" << (entry).num_searched << sync_endl;\
		}\
        if ((entry).hash_high == DEBUG_CH_HASH_0 >> 32) {\
            sync_cout << "debug[ch]: pn=inf at UpdateDnZero()" << ",id=" << threadid << ",pn=" << (entry).pn << ",dn=" << (entry).dn << ",depth=" << (entry).depth << ",hand=" << (entry).hand.value() << ",num_searched=" << (entry).num_searched << sync_endl;\
		}\
\
        (entry).pn = (pn_inf);\
        (entry).dn = 0;\
        if (should_update_hand<DnIsZero>(new_hand_moveable, (entry).hand)) {\
        	(entry).hand = std::move(new_hand_moveable);\
        }\
    }\
    /* 最大手数引き分け不詰みは反証駒が利用できず、任意の純粋な不詰みより汎用性が低いので確定で上書き。*/\
    else if ((entry).dn == 0 && (entry).num_searched == REPEAT) {\
        if ((entry).hash_high == DEBUG_HASH_0 >> 32) {\
            sync_cout << "debug: pn=inf(set ns=1) at UpdateDnZero()" << ",id=" << threadid << ",pn=" << (entry).pn << ",dn=" << (entry).dn << ",depth=" << (entry).depth << ",hand=" << (entry).hand.value() << ",num_searched=" << (entry).num_searched << sync_endl;\
		}\
        if ((entry).hash_high == DEBUG_CH_HASH_0 >> 32) {\
            sync_cout << "debug[ch]: pn=inf(set ns=1) at UpdateDnZero()" << ",id=" << threadid << ",pn=" << (entry).pn << ",dn=" << (entry).dn << ",depth=" << (entry).depth << ",hand=" << (entry).hand.value() << ",num_searched=" << (entry).num_searched << sync_endl;\
		}\
\
        /* REPEAT 以外の何かしらの値に書き換える。0 はダメだと思う。1 なら問題ないんじゃないかな。。。*/\
        (entry).num_searched = 1;\
        if (should_update_hand<DnIsZero>(new_hand_moveable, (entry).hand)) {\
        	(entry).hand = std::move(new_hand_moveable);\
        }\
	}\
    /* pn の方にset されるinf には1種類しか無いので、hand だけ更新。*/\
    else if (should_update_hand<DnIsZero>(new_hand_moveable, (entry).hand)) {\
        /* debug */\
        if ((entry).hash_high == DEBUG_HASH_0 >> 32) {\
            sync_cout << "debug: pn=inf(when dn==0) at UpdateDnZero()" << ",id=" << threadid << ",pn=" << (entry).pn << ",dn=" << (entry).dn << ",depth=" << (entry).depth << ",hand=" << (entry).hand.value() << ",num_searched=" << (entry).num_searched << sync_endl;\
		}\
        if ((entry).hash_high == DEBUG_CH_HASH_0 >> 32) {\
            sync_cout << "debug[ch]: pn=inf(when dn==0) at UpdateDnZero()" << ",id=" << threadid << ",pn=" << (entry).pn << ",dn=" << (entry).dn << ",depth=" << (entry).depth << ",hand=" << (entry).hand.value() << ",num_searched=" << (entry).num_searched << sync_endl;\
		}\
\
    	(entry).hand = std::move(new_hand_moveable);\
    }\
} while (0);

// TODO: pn == 0 の場合も優越局面等はnum_searched = REPEAT なので、UpdateDnZero の場合と同様の処理をしてやる。
#define UpdatePnZero(entry, dn_inf, new_hand_moveable)\
do {\
    /* pn!=0 の場合そもそも他の局面で使える情報が無いので汎用性は皆無。pn=0とする方が必ず汎用性が高い。*/\
    if ((entry).pn != 0) {\
    	(entry).pn = 0;\
    	(entry).dn = (dn_inf);\
    	if (should_update_hand<PnIsZero>(new_hand_moveable, (entry).hand)) {\
    	    (entry).hand = std::move(new_hand_moveable);\
        }\
    }\
    else if (should_update_hand<PnIsZero>(new_hand_moveable, (entry).hand)) {\
    	(entry).hand = std::move(new_hand_moveable);\
    	(entry).dn = (dn_inf);\
    }\
} while (0);

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
static const constexpr int kInfinitePnDn = 100000000;

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
		while (true) {
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
template<LookUpType lut>
TTEntry& TranspositionTable::LookUp(const Key key, const Hand hand, const uint16_t depth) {
	auto& entries = tt[key & clusters_mask];    // TODO: これデータ競合あるよね。
	uint32_t hash_high = key >> 32;

#ifdef DEBUG
	if (hash_high == HASH_HIGH_DEBGU_TARGET3) {
		std::cout << "[" << key << "]" << std::endl;
	}
#endif

	return LookUpDirect<lut>(entries, hash_high, hand, depth);
}

// TODO
//     : 一応、entries にアクセスする為のkey でposition_mutex 取得すればスレッドセーフだが、
//       LookUpDirect 自体は、entries に置換表からアクセスする為のkey を知らないのでこの関数ではlock出来ない。
// NOTE
//     : 引数hand には、LookUp<bool or_node>() を通じてOR なら手番側の持ち駒が、AND には敵側の持ち駒が渡される(はず)。
//       -> 要するに、hand は詰ます側の持ち駒が渡される。
template<LookUpType lut>
TTEntry& TranspositionTable::LookUpDirect(
	Cluster& entries, const uint32_t hash_high, const Hand hand, const uint16_t depth
) {
	if constexpr (lut == LookUpTypeNum) {
		std::runtime_error("LookUpDirect() got unexpected LookUpType.");
	}

	int max_pn = 1;
	int max_dn = 1;

	// 検索条件に合致するエントリを返す
	for (size_t i = 0; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
		TTEntry& entry = entries.entries[i];

		entry.lock();

		// NOTE
		//     : 今の世代に一致しないものは、ゴミと見なす？(利用せずに上書きする？)
		if (generation != entry.generation) {


			if (hash_high == DEBUG_HASH_0 >> 32) {
			    sync_cout
					<< "debug: LookUp[new]" << ",max_pn=" << max_pn << ",max_dn=" << max_dn
					<< ",depth=" << depth << ",hand=" << hand.value() << sync_endl;
			}

			if (hash_high == DEBUG_CH_HASH_0 >> 32) {
				sync_cout
					<< "debug[ch]: LookUp[new]" << ",max_pn=" << max_pn << ",max_dn=" << max_dn
					<< ",depth=" << depth << ",hand=" << hand.value() << sync_endl;
			}

			// NOTE: ここは全く問題無いはず。。。
			// 空のエントリが見つかった場合
			entry.hash_high = hash_high;
			entry.depth = depth;
			entry.hand = hand;
			// TODO
			//     : 最悪の場合を想定してmax を登録しておく？
			//       だとしたら途中までじゃなくてなんで全体の最大値にしないの？この辺りは雑で良いってこと？
			//     : 証明のしづらさ(pn, dn 共に)を不当に高く見積もってしまう気がしている。
			//       これは無くすか、max の半分とかにするべきでは？(pn = (max_pn / 2) + 1 とする。0にならないようにね。)
			// NOTE
			//     : ここ、max_pn == inf. max_dn == 1(初期値) が世に送り出される可能性ある。
			//       これ、lu == ReadOnly (つまりdlshogi の実装) のままなら、dn == 0 or pn == 0 の時点で
			//       その条件節の中に入り込むので、max_pn = とかmax_dn = の処理の条件に入る事はない。
			//       それゆえ、max_pn == inf とかmax_dn == inf みたいな事に張らなかった。
			entry.pn = max_pn;
			entry.dn = max_dn;
			entry.generation = generation;
			entry.num_searched = 0;

			entry.unlock();
			return entry;

		}

		// NOTE
		//     : より汎用性の高い局面である条件は、
		//       1. pp/dp がある関係を満たす
		//       2. num_searched != REPEATs
		if (hash_high == entry.hash_high && generation == entry.generation) {    // NOTE: 同じ局面と見なす必要条件
			if (hand == entry.hand && depth == entry.depth) {

				entry.unlock();
				// keyが合致するエントリを見つけた場合
				// 残りのエントリに優越関係を満たす局面があり証明済みの場合、それを返す
				for (i++; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
					TTEntry& entry_rest = entries.entries[i];
					entry_rest.lock();
					if (generation != entry_rest.generation) {
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
								if (hash_high == DEBUG_HASH_0 >> 32) {
									sync_cout
										<< "debug: LookUp[match depth]" << ",pn=" << entry_rest.pn << ",dn=" << entry_rest.dn
										<< ",depth=" << entry_rest.depth << ",hand=" << entry_rest.hand.value()
										<< ",num_searched=" << entry_rest.num_searched << ",query_hand=" << hand.value() << sync_endl;
								}
								if (hash_high == DEBUG_CH_HASH_0 >> 32) {
									sync_cout
										<< "debug[ch]: LookUp[match depth]" << ",pn=" << entry_rest.pn << ",dn=" << entry_rest.dn
										<< ",depth=" << entry_rest.depth << ",hand=" << entry_rest.hand.value()
										<< ",num_searched=" << entry_rest.num_searched << ",query_hand=" << hand.value() << sync_endl;
								}
								entry_rest.unlock();
								return entry_rest;
							}
						}
						else if (entry_rest.dn == 0) {    // NOTE: 不詰み証明済み
							if (entry_rest.hand.isEqualOrSuperior(hand) && entry_rest.num_searched != REPEAT) {
								if (hash_high == DEBUG_HASH_0 >> 32) {
									sync_cout
										<< "debug: LookUp[match depth]" << ",pn=" << entry_rest.pn << ",dn=" << entry_rest.dn
										<< ",depth=" << entry_rest.depth << ",hand=" << entry_rest.hand.value()
										<< ",num_searched=" << entry_rest.num_searched << ",query_hand=" << hand.value() << sync_endl;
								}
								if (hash_high == DEBUG_CH_HASH_0 >> 32) {
									sync_cout
										<< "debug[ch]: LookUp[match depth]" << ",pn=" << entry_rest.pn << ",dn=" << entry_rest.dn
										<< ",depth=" << entry_rest.depth << ",hand=" << entry_rest.hand.value()
										<< ",num_searched=" << entry_rest.num_searched << ",query_hand=" << hand.value() << sync_endl;
								}
								entry_rest.unlock();
								return entry_rest;
							}
						}
					}
					entry_rest.unlock();
				}
   
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
			// 優越関係を満たす局面に証明済みの局面がある場合、それを返す
			if constexpr (lut == ReadOnly) {
				if (entry.pn == 0) {
					if (hand.isEqualOrSuperior(entry.hand) && entry.num_searched != REPEAT) {
						if (hash_high == DEBUG_HASH_0 >> 32) {
							sync_cout
								<< "debug: LookUp[not match depth]" << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value()
								<< ",num_searched=" << entry.num_searched << ",query_hand=" << hand.value() << sync_endl;
						}
						if (hash_high == DEBUG_CH_HASH_0 >> 32) {
							sync_cout
								<< "debug[ch]: LookUp[not match depth]" << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value()
								<< ",num_searched=" << entry.num_searched << ",query_hand=" << hand.value() << sync_endl;
						}
						entry.unlock();
						return entry;
					}
				}
				else if (entry.dn == 0) {
					if (entry.hand.isEqualOrSuperior(hand) && entry.num_searched != REPEAT) {
						if (hash_high == DEBUG_HASH_0 >> 32) {
							sync_cout
								<< "debug: LookUp[not match depth]" << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value()
								<< ",num_searched=" << entry.num_searched << ",query_hand=" << hand.value() << sync_endl;
						}
						if (hash_high == DEBUG_CH_HASH_0 >> 32) {
							sync_cout
								<< "debug[ch]: LookUp[not match depth]" << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value()
								<< ",num_searched=" << entry.num_searched << ",query_hand=" << hand.value() << sync_endl;
						}
						entry.unlock();
						return entry;
					}
				}
				// TODO
				//     : 何故以下の場合に最大値を更新するのか？
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
			else if constexpr (lut == UpdateOnly) {
				// TODO
				//     : 何故以下の場合に最大値を更新するのか？
				if (entry.pn < kInfinitePnDn && entry.hand.isEqualOrSuperior(hand)) {
					if (entry.pn > max_pn) {
						max_pn = entry.pn;
					}
				}
				else if (entry.dn < kInfinitePnDn && hand.isEqualOrSuperior(entry.hand)) {
					if (entry.dn > max_dn) {
						max_dn = entry.dn;
					}
				}
			}
		}
		entry.unlock();
	}

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

template <bool or_node, LookUpType lut>
TTEntry& TranspositionTable::LookUp(const Position& n) {
	// NOTE: このLookUp<>() はLookUp() のラッパー
	auto& retVal = LookUp<lut>(n.getBoardKey(), or_node ? n.hand(n.turn()) : n.hand(oppositeColor(n.turn())), n.gamePly());
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
template <bool or_node, LookUpType lut>
TTEntry& TranspositionTable::LookUpChildEntry(const Position& n, const Move move) {
	Cluster* entries;
	uint32_t hash_high;
	Hand hand;
	//const auto&& key = GetChildFirstEntry<or_node>(n, move, entries, hash_high, hand);
	GetChildFirstEntry<or_node>(n, move, entries, hash_high, hand);

	//std::cout << "key = [" << u64(key) << "]" << std::endl;
	return LookUpDirect<lut>(*entries, hash_high, hand, n.gamePly() + 1);
}

// TODO: やねうら王みたいに、最初から物理メモリ上に確保したい
// 領域を再確保することでreset してる。非推奨。
[[deprecated("Use Clear() instead.")]]
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

void TranspositionTable::Clear(int n_threads) {
	if (tt_raw) {
		if (n_threads == 1) {
			sync_cout
				<< "info string clearing hash = "
				<< (sizeof(Cluster) * (num_clusters / 1024)) / 1024 << "mb"
				<< ", threads = " << n_threads << sync_endl;
			std::memset(tt_raw, 0, num_clusters * sizeof(Cluster) + CacheLineSize);
		}
		else {
			sync_cout
				<< "info string Error!: TT::Clear() only supports clearing with 1 thread, given n_threads = " << n_threads << sync_endl;
			throw std::runtime_error("TT::Clear() only supports clearing with 1 thread.");
		}
	}
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
	//     : b --(*1024)-> kb -- (*1024)-> mb
	//       sizeof() の単位byte に合わせる為に2**20 を掛ける。
	int64_t new_num_clusters = 1LL << msb((hash_size_mb * 1024 * 1024) / sizeof(Cluster));
	if (new_num_clusters == num_clusters) {
		return;
	}

	num_clusters = new_num_clusters;
	// TOOD
	//     : これ、普通にif で判定すべきな気はする。
	//       現状、hash_size_mb = 512mb 未満だとアウト。恐らくね。
	//       -> えっと、今、pos_mutex 使ってます？使ってないですよね？これ要らなく無いですか？
	assert(num_clusters >= POS_MUTEX_NUM && "expected num_clusters >= POS_MUTEX_NUM, but got num_clusters < POS_MUTEX_NUM");

	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}

	// NOTE: OverFlow とか桁落ちとかを防ぐために、割って掛けて割ってる
	sync_cout << "info string alloc_size=" << (sizeof(Cluster) * (new_num_clusters / 1024)) / 1024 << "mb" << sync_endl;

	// NOTE
	//     : 特に型は指定せずに、ただただ(new_num_clusters * sizeof(Cluster) + CacheLineSize) * (1) byte の領域を確保
	// TODO
	//     : CacheLineSize とは？
	//     : ((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1)) って何してる？
	//       -> 実験してみた感じあれやな、TT の先頭アドレスをCacheLineSize の自然数倍の位置に来るように調整してるっぽい。
	//          ほんでもって、これをするために、calloc ではCacheLineSize 分だけ多めに確保してるっぽい
	//          (先頭の位置をスライドできるだけの余裕を持たせる)。
	tt_raw = std::calloc(new_num_clusters * sizeof(Cluster) + CacheLineSize, 1);
	if (tt_raw == nullptr) {
		// なんかbad_alloc 出なかったのでこれで。
		sync_cout << "info string Error: bad_alloc! Resize() failed to calloc" << sync_endl;
		exit(1);
	}

	tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));
	clusters_mask = num_clusters - 1;

	if (clusters_mask > CLUSTERS_MASK_LIMIT) {
		// cluster_size==512 なら、sizeof(Cluster)==16384 なので、現状hash_size==64TB (2**26mb) まで耐えるはず。
		// ((2**26) * 1024 * 1024 / 16384 == 2**32)
		sync_cout
			<< "info string Error!: hash_size is too large and cluster_mask exceeds the upper limit."
			<< sync_endl;
	}
}

void TranspositionTable::NewSearch() {
	++generation;
	// TODO: なぜこうするの？初めはgeneration = -1 ってことかい？
	if (generation == 0) generation = 1;
}

// NOTE
//     : 近接王手ならtrue
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
template <bool or_node, bool shared_stop, bool is_root>
void ParallelDfPn::dfpn_inner(
	Position& n, const int thpn, const int thdn/*, bool inc_flag*/, const uint16_t maxDepth, int64_t& searchedNode, const int threadid
) {
	auto& entry = transposition_table->LookUp<or_node, UpdateOnly>(n);


	if (n.getBoardKey() == DEBUG_HASH_PRINT_0) {
		std::ostringstream oss_for_pos_print;
		n.print(oss_for_pos_print);
		sync_cout << "[DEBUG_HASH_PRINT_0:\n" << oss_for_pos_print.str() << sync_endl;
	}

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
			//     : まぁ他でも、一度出た結論はひっくり返してはならない。(はず)
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				// NOTE
				//     : 既に不詰み証明済みの場合、最大手数不詰みは、不詰みの中でも最弱なので、
				//       汎用性は横ばいか低下することになるので、何もしなくて良い。
				if (entry.dn != 0) {
					// TODO20231026: 上書きのみ
					// NOTE: この場合、反証駒も何もない。だって最大手数引き分けなんだもの。
					
					// debug
				    if ((entry).hash_high == DEBUG_HASH_0 >> 32) {
				        sync_cout << "debug: pn=inf at first, maxmoves" << sync_endl; 
				    }
					if ((entry).hash_high == DEBUG_CH_HASH_0 >> 32) {
						sync_cout << "debug[ch]: pn=inf at first, maxmoves" << sync_endl;
					}

					entry.pn = kInfinitePnDn;
					entry.dn = 0;
					entry.num_searched = REPEAT;    // TODO: REPEAT って何？
				}

			}
			entry.unlock();
			return;    // NOTE: serach_result = 中断(depth limit)
			}
		}

	//volatile const std::string sfen = n.toSFEN();

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
				// 反証駒
#ifdef DFPN_MOD_V7
				// NOTE
				//     : 現状の情報では、先手側の持ち駒以上に強いことは言えないはずだし、
				//       この局面で先手が先手の今の持ち駒を持っていて詰まないことは揺るがない事実であり、適切なはず。。。
				Hand new_hand(n.hand(n.turn()));
#else
				// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
				Hand new_hand(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
#endif

				UpdateDnZero(entry, kInfinitePnDn, new_hand);
			}
		}
		else {
#ifdef DFPN_MOD_V3
			if (entry.dn != 0) {
#else
			if (true) {
#endif
			    // TODO: ここって、証明駒=0 ってset しても良いのでは？
				// 相手の手番でここに到達した場合は王手回避の手が無かった、
				// 1手詰めを行っているため、ここに到達することはない
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
				auto& entry2 = transposition_table->LookUp<false, UpdateOnly>(n);

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
					//     : dn == inf + 3: 2手詰め。
					//       dn == inf + 2: 1手詰め。
					//       dn == inf + 1: 0手詰め。


					entry2.lock();
#ifdef DFPN_MOD_V3
					if (entry2.dn != 0) {
#else
					if (true) {
#endif
						// 完全に
						entry2.pn = 0;
						entry2.dn = kInfinitePnDn + 1;

						// 以下は合法だと思うが、無意味だと思う。だって完全に詰みの局面だし、親のentry でもset するから。
						//entry2.hand.set(0);
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
						// NOTE
						//     : 定義通りにするなら、詰んだ局面であるentry2 での証明駒 = 0 で、
						//       m が駒を打つ手ならば "entry での証明駒 = entry.hand + 打った駒" と差分更新
						// 証明駒を初期化
						Hand new_hand(0);
						// 打つ手ならば証明駒に加える
						if (m.isDrop()) {
							new_hand.plusOne(m.handPieceDropped());
						}

						// new_hand は0 で初期化してるので、取る手の場合に1引く ことを考える必要はない。

						// TODO20231028
						//     : OrNode だからこういういの要らんのでは？AndNode だけじゃないの？
						//       -> まぁ大きくしてるだけだと思うから、不当に小さくしない限りはバグにはならないはず。。。
						//       -> いや、反証駒として利用されたらダメ。だけど、そんなことってあるんかいな？
						// 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
						if (!moveGivesNeighborCheck(n, m)) {    // 
							new_hand.setPP(n.hand(n.turn()), n.hand(oppositeColor(n.turn())));
						}

						// dn == inf + 2: 1手詰め。
						UpdatePnZero(entry, kInfinitePnDn + 2, new_hand);
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
					// TODO20231028
					//     : is_update_ok でLookUp してるのでdepth は一致してるはずで、
					//       別depthでは詰みの局面が最大手数引き分けによる不詰みに書き換えられる心配は無いはず。。。
#ifdef DFPN_MOD_V2
					if (entry2.pn != 0) {
#else
					if (true) {
#endif
						if (entry2.dn != 0) {
							// debug
							if ((entry2).hash_high == DEBUG_HASH_0 >> 32) {
								sync_cout << "debug: pn=inf at maxmoves" << sync_endl;
							}
							if ((entry2).hash_high == DEBUG_CH_HASH_0 >> 32) {
								sync_cout << "debug[ch]: pn=inf at maxmoves" << sync_endl;
							}
							entry2.pn = kInfinitePnDn;
							entry2.dn = 0;
							entry2.num_searched = REPEAT;
						}
					}
					entry2.unlock();

					continue;
					}

				// NOTE
				//     : 今は遷移してAndNode に居るので、一つでも詰みを逃れる指し手があれば、NEXT_CHECK へとジャンプする。
				// TODO
				//     : 1手詰めだから、その持ち駒が必要か否かのフラグでも良いかも。わざわざu32 使う必要ないのでは。
				// 子局面の証明駒の和集合
				u32 pps[HandPieceNum] = {0};

				const CheckInfo ci2(n);
				for (const auto& move : move_picker2) {    // NOTE: AndNode での指し手を探索
					const Move& m2 = move.move;

					// この指し手で逆王手になるなら、不詰めとして扱う
					if (n.moveGivesCheck(m2, ci2)) {
						goto NEXT_CHECK;
					}

					n.doMove(m2, si2, ci2, false);

					const auto&& mate_move_in_1ply = n.mateMoveIn1Ply<mmin1ply_Additional>();
					if (mate_move_in_1ply) {
						auto& entry1 = transposition_table->LookUp<true, UpdateOnly>(n);
						entry1.lock();
#ifdef DFPN_MOD_V3
						if (entry1.dn != 0) {
#else
						if (true) {
#endif
#ifdef DFPN_MOD_V6
							Hand new_hand(entry1.hand);

#else
							// 証明駒を初期化
							Hand new_hand(0);
							// 打つ手ならば証明駒に加える
							if (mate_move_in_1ply.isDrop()) {
								const HandPiece&& hp = mate_move_in_1ply.handPieceDropped();
								new_hand.plusOne(hp);
#if 1
								// NOTE
								//     : 1手詰め局面なので、子局面の証明駒の和集合に含まれる、
								//       任意の種類の駒の最大値は1である。
								//       つまり、比較も無しに1 に相当する値を代入してよい。
								//     : new_hand には直前でplusOne した駒以外は登録されてないので、
								//       exists<hp>() と.value() は等しい。
								pps[hp] = new_hand.value();
#else
								const u32 child_hp_exists = new_hand.exists<hp>();
								if (pps[hp] < child_hp_exists) pps[hp] = child_hp_exists;
#endif
							}

							// new_hand は0 で初期化してるので、取る手の場合に1引く ことを考える必要はない。

#endif
							// TODO20231028
							//     : 新しいentry の場合、entry.hand == 0 なので、should_update_hand() はfalse となってしまう。
							//       ここで初めてpn == 0 となるなら、証明駒は必ずset すべき。
							//       既にpn == 0 であるなら、以下のままでよい。
							//       -> entry.pn != 0 ならすぐに問答無用でentry.hand.set(new_hand) して、
							//          entry.pn == 0 なら、より汎用性が高くなる時のみ(should_update_hand == true) 更新する。
							//     : この考えを元に、これまでの奴も正しくない奴があれば修正せよ。
							UpdatePnZero(entry1, kInfinitePnDn + 2, new_hand)


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
				//// すべて詰んだ

				entry2.lock();
#ifdef DFPN_MOD_V3
				if (entry2.dn != 0) {
#else
				if (true) {
#endif
#ifdef DFPN_MOD_V6
					Hand new_hand(entry2.hand);
#else
					Hand new_hand(0);
					new_hand.set(pps[HPawn] | pps[HLance] | pps[HKnight] | pps[HSilver] | pps[HGold] | pps[HBishop] | pps[HRook]);

					// テーブルの参照で終わるので遅くはないはず。(最後に& はするけど2回だけ。)
					// 遠隔王手なら、後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
					if (!(n.checkersBB() & n.attacksFrom<King>(n.kingSquare(n.turn()))
						|| n.checkersBB() & n.attacksFrom<Knight>(n.turn(), n.kingSquare(n.turn())))
						) {
						new_hand.setPP(n.hand(oppositeColor(n.turn())), n.hand(n.turn()));
					}
#endif
					// dn == inf + 3: 2手詰め。
					UpdatePnZero(entry2, kInfinitePnDn + 3, new_hand);
				}

				n.undoMove(m);

				entry.lock();
#ifdef DFPN_MOD_V3
				if (entry.dn != 0) {
#else
				if (true) {
#endif
					// OrNode での証明駒は、正解手と子ノードの証明駒 から差分計算する。
					// 差分計算した奴と、現状のhand 、どっちが強い方を採用する(UpdatePnZero にて)。
					// NOTE
					//     : これ、dlshogiはnew_hand(entry.hand) としてる、即ち打ったり取ったりしないなら先手の持ち駒をそのまま証明駒としてる。
					//       けど、(打つ, 取る, それ以外 のどれであれ)子ノードの証明駒から算出した奴と、現状の先手の持ち駒を比べて、
					//       より汎用性の高い方をset すべきだと僕は思う。
#ifdef DFPN_MOD_V6
					Hand new_hand(entry.hand);
#else
					Hand new_hand(entry2.hand);
					if (m.isDrop()) {
						const HandPiece& hp = m.handPieceDropped();
						new_hand.plusOne(hp);    // 差分更新
					}
					else {
						const Piece& to_pc = n.piece(m.to());
						if (to_pc != Empty) {    // NOTE: 駒を取る手である
							const PieceType& pt = pieceToPieceType(to_pc);
							const HandPiece& hp = pieceTypeToHandPiece(pt);
							if (new_hand.exists(hp)) {
								new_hand.minusOne(hp);    // 差分更新
							}
						}
					}
#endif

					UpdatePnZero(entry, kInfinitePnDn, new_hand);
				}
				entry2.unlock();
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
					// NOTE
					//     : 2手詰め局面とかはnum_searched が0 のままだったりするはずで、そういう局面を上書きしないように。。。
					if (entry2.pn != 0 && entry2.dn != 0) {
						entry2.num_searched = 1;
						entry2.pn = static_cast<int>(move_picker2.size());
						entry2.dn = static_cast<int>(move_picker2.size());
					}
				}
				entry2.unlock();
			}
		}
		else {
			// 2手読みチェック
			StateInfo si2;
			// この局面ですべてのevasionを試す
			// 子局面の証明駒の和集合
			u32 pps[HandPieceNum] = { 0 };
			const CheckInfo ci2(n);
			for (const auto& move : move_picker) {    // NOTE: AndNode の指し手で、全部調べる。
				const Move& m2 = move.move;

				// この指し手で逆王手になるなら、不詰めとして扱う
				if (n.moveGivesCheck(m2, ci2)) {
					goto NO_MATE;
				}

				n.doMove(m2, si2, ci2, false);

				// TODO
				//     : dlshogi にならって、
				//       template <bool Additional = true> Move mateMoveIn1Ply(); を
				//       template <bool Additional = false> Move mateMoveIn1Ply(); に。
				//       (正確には、dlshogi はAdditional が無い。)
				if (const Move move = n.mateMoveIn1Ply<mmin1ply_Additional>()) {
					auto& entry1 = transposition_table->LookUp<true, UpdateOnly>(n);

					entry1.lock();
#ifdef DFPN_MOD_V3
					if (entry1.dn != 0) {
#else
					if (true) {
#endif

						Hand new_hand(0);
						if (move.isDrop()) {
							new_hand.plusOne(move.handPieceDropped());
						}
						
						// new_hand は0 で初期化してるので、取る手の場合に1引く ことを考える必要はない。

						// TODO20231028
						//     : OrNode だからこういういの要らんのでは？AndNode だけじゃないの？
						// 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
						if (!moveGivesNeighborCheck(n, move)) {
							new_hand.setPP(n.hand(n.turn()), n.hand(oppositeColor(n.turn())));
						}

#ifdef DFPN_MOD_V6
						// 何もなし
#else
						// NOTE: 証明駒の和集合を求める。
						for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
							const u32 child_hp_exists = new_hand.exists(hp);
							if (child_hp_exists > pps[hp]) pps[hp] = child_hp_exists;
						}
#endif
						UpdatePnZero(entry1, kInfinitePnDn + 2, new_hand);
					}
					entry1.unlock();
				}
				else {
					// NOTE
					//     : m2 を指した時に、先手側に詰ます手が無い。
					//       この時、m2 を指すことで2手以内の詰みは逃れている。
					//     : 特に、この局面で先手側に指し手が無いなら、完全に不詰みである(何手あっても詰まない)。
					MovePicker<true> move_picker2(n);
					if (move_picker2.empty()) {
						auto& entry1 = transposition_table->LookUp<true, UpdateOnly>(n);
						entry1.lock();
#ifdef DFPN_MOD_V2
						if (entry1.pn != 0) {
#else
						if (true) {
#endif
							// 反証駒
#ifdef DFPN_MOD_V7
							Hand new_hand(n.hand(n.turn()));
#else
							// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
							Hand new_hand(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
#endif

							UpdateDnZero(entry1, kInfinitePnDn, new_hand);
						}

						n.undoMove(m2);    // TODO: undoMove() 重いならentry1 のクリティカルセクションからどかす。
						entry.lock();
#ifdef DFPN_MOD_V2
						if (entry.pn != 0) {
#else
						if (true) {
#endif
							// TODO20231028
							//     : https://tadaoyamaoka.hatenablog.com/entry/2018/05/31/001822
							//       反証駒の実装がまだ理解できてないけど、ひとまず受け入れて先に進む。
							//       -> だいたい理解出来た。
							//          後手番が駒を取った場合に反証駒に追加するのは厳密には正しくない。
							//          何故なら、盤上の駒を取る場合もあり、その駒が必ずしも先手の持ち駒を起源とするとは限らないからである。
							//          そこでOrNode で毎度、先手が一枚も持ってないのに反証駒となってる駒は、盤上の駒であるとして削除して、
							//          正しさを保っている(本来の目的がどうかは知らんが、実験したらそれによってOrNode で正しく訂正されていた。)。
							//       -> つまり、恐らくだけど、反証駒はAndNode においては間違った値となる可能性があると思う。
							//       -> 反証駒も先手の駒であり、反証駒が実際に使われるのはOrNode の時であり、
							//          OrNode において正しければokということ？
							//          (AndNode における反証駒はOrNode に反証駒を伝搬するための物でしかないから、ok 的な？)
							//          若しくは、ルートへと誤差が伝搬さえせず、毎度OrNode で誤差を堰き止めてくれればそれでok ということ？
							//          (AndNode で参照した場合、優越関係がおかしくなる場合があるはずなんだよな～。これ許容しちゃだめだよな～。)
							Hand new_hand(entry1.hand);
#ifdef DFPN_MOD_V7
							if (m2.isDrop()) {
								const HandPiece& hp = m2.handPieceDropped();
								const int& dp_hp_cnt = new_hand.numOf(hp);
								const int& actual_total_hp_cnt = n.hand(n.turn()).numOf(hp) + n.hand(oppositeColor(n.turn())).numOf(hp);
								// 反証駒にある駒hp が、実際の先手の持ち駒と同じかそれ以上持っていた場合、m2 を打てない。
								if (actual_total_hp_cnt <= dp_hp_cnt) {
									// 後手が持ち駒から打てるように、actual_total_hp_cnt - 1個 にset する。
									new_hand.minusN(hp, dp_hp_cnt);
									new_hand.plusN(hp, actual_total_hp_cnt - 1);
								}
							}
#else
							// 子局面の反証駒を設定
							// 打つ手ならば、反証駒から削除する
							if (m2.isDrop()) {
								const HandPiece& hp = m2.handPieceDropped();
								if (new_hand.exists(hp)) {
									new_hand.minusOne(hp);
								}
							}
							// 先手の駒を取る手ならば、反証駒に追加する
							else {
								const Piece& to_pc = n.piece(m2.to());
								if (to_pc != Empty) {
									const PieceType& pt = pieceToPieceType(to_pc);
									const HandPiece& hp = pieceTypeToHandPiece(pt);
									new_hand.plusOne(hp);
								}
							}
#endif

							// TODO
							//     : ここ、これじゃだめかもな。
							//       -> え、何故？(by 未来の私。)
							UpdateDnZero(entry, kInfinitePnDn, new_hand);
						}
						entry1.unlock();
						entry.unlock();    // TODO: ここのunlock も要らん気もするな。
						return;
					}
					n.undoMove(m2);
					goto NO_MATE;
				}

				n.undoMove(m2);
			}
			//// すべて詰んだ
			entry.lock();
#ifdef DFPN_MOD_V3
			if (entry.dn != 0) {
#else
			if (true) {
#endif
#ifdef DFPN_MOD_V6
				Hand new_hand(entry.hand);
#else
				Hand new_hand(0);
				new_hand.set(pps[HPawn] | pps[HLance] | pps[HKnight] | pps[HSilver] | pps[HGold] | pps[HBishop] | pps[HRook]);

				// テーブルの参照で終わるので遅くはないはず。(最後に& はするけど2回だけ。)
				// 遠隔王手なら、後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
				if (!(n.checkersBB() & n.attacksFrom<King>(n.kingSquare(n.turn()))
					|| n.checkersBB() & n.attacksFrom<Knight>(n.turn(), n.kingSquare(n.turn())))
					) {
					new_hand.setPP(n.hand(oppositeColor(n.turn())), n.hand(n.turn()));
				}
#endif
				// dn == inf + 3: 2手詰め。
				UpdatePnZero(entry, kInfinitePnDn + 3, new_hand);
			}
			entry.unlock();
			return;

		NO_MATE:;
		}
	}
	else {
		entry.unlock();
	}

	// NOTE
	//     : どうしてこのタイミングで千日手判定するんや？
	//       -> 恐らくやけど、dfpn 呼ぶ局面では詰みの方が珍しいからやないかな。
	//          (先に短手数の詰みを確認しておいた方が、この関数を抜けると期待される時間は小さくなる。)
	//     : 引き分けなら評価値を書き換えるやつ、
	//       例えばdepth=6とdepth=9 で同じ局面が出てきた場合、depth=6 の詰み不詰みも千日手として値が書き換えられるが、
	//       qhapaq の記事のように、千日手は入り口から一番遠い局面で初めて不詰み扱いしなければならないので、
	//       depth=6 では不詰みとしてはいけない(はず)。
	// TODO
	//     : 何故RepInf は無いの？
	// 千日手のチェック
	// NOTE
	//     : 証明駒、反証駒をどうすれば良いのかあんま分かってない。
	//       -> あれやね、この局面は証明駒、反証駒が難しいから、
	//          num_searched = REPEAT として、この時の情報は無視するようにしてるんやね。
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
				if (entry.pn != 0) {
					entry.pn = 0;
					entry.dn = kInfinitePnDn;
					entry.num_searched = REPEAT;
				}
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
				if (entry.dn != 0) {
					// debug
					if ((entry).hash_high == DEBUG_HASH_0 >> 32) {
						sync_cout
							<< "pn=inf at [And RepWin]:id = " << threadid << ", ply=" << n.gamePly()
							<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
							<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << sync_endl;
					}
					// debug
					if ((entry).hash_high == DEBUG_CH_HASH_0 >> 32) {
						sync_cout
							<< "[ch]pn=inf at [And RepWin]:id = " << threadid << ", ply=" << n.gamePly()
							<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
							<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << sync_endl;
					}

					entry.pn = kInfinitePnDn;
					entry.dn = 0;
					entry.num_searched = REPEAT;
				}
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
				if (entry.dn != 0) {
					// debug
					if ((entry).hash_high == DEBUG_HASH_0 >> 32) {
						sync_cout
							<< "pn=inf at [Or RepLose]:id = " << threadid << ", ply=" << n.gamePly()
							<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
							<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << sync_endl;
					}
					if ((entry).hash_high == DEBUG_CH_HASH_0 >> 32) {
						sync_cout
							<< "[ch]pn=inf at [Or RepLose]:id = " << threadid << ", ply=" << n.gamePly()
							<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
							<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << sync_endl;
					}

					entry.pn = kInfinitePnDn;
					entry.dn = 0;
					entry.num_searched = REPEAT;
				}
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
				if (entry.pn != 0) {
					entry.pn = 0;
					entry.dn = kInfinitePnDn;
					entry.num_searched = REPEAT;
				}
			}
			entry.unlock();
			}
		return;

	case RepetitionDraw:
		// TODO
		//     : 何故ここは通らないはず？
		//       -> あ、あれか、詰将棋やから、王手の絡まない千日手は起きない、
		//          即ち 優勢局面と劣勢局面を除けばRepWin or RepLose しか起きへんのか。
		// 普通の千日手
		// ここは通らないはず
		entry.lock();
#ifdef DFPN_MOD_V2
		if (entry.pn != 0) {
#else
		if (true) {
#endif
			if (entry.dn != 0) {
				// debug
				if ((entry).hash_high == DEBUG_HASH_0 >> 32) {
					sync_cout
						<< "pn=inf at [RepDraw]:id = " << threadid << ", ply=" << n.gamePly()
						<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
						<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << sync_endl;
				}
				if ((entry).hash_high == DEBUG_CH_HASH_0 >> 32) {
					sync_cout
						<< "[ch]pn=inf at [RepDraw]:id = " << threadid << ", ply=" << n.gamePly()
						<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
						<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << sync_endl;
				}

				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				entry.num_searched = REPEAT;
			}
		}
		else {

			// debug
			if ((entry).hash_high == DEBUG_HASH_0 >> 32) {
				sync_cout
					<< "[RepDraw] but already pn==0:id = " << threadid << ", ply=" << n.gamePly()
					<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
					<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << sync_endl;
			}
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
				if (entry.dn != 0) {
					// debug
					if ((entry).hash_high == DEBUG_HASH_0 >> 32) {
						sync_cout
							<< "pn=inf at [And RepSup]:id = " << threadid << ", ply=" << n.gamePly()
							<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
							<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << sync_endl;
					}
					if ((entry).hash_high == DEBUG_CH_HASH_0 >> 32) {
						std::ostringstream oss;
						n.print(oss);
						sync_cout
							<< "[ch]pn=inf at [And RepSup]:id = " << threadid << ", ply=" << n.gamePly()
							<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
							<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << "\n" << oss.str() << std::endl;
						n.print_si_history();
						std::cout << IO_UNLOCK;
					}

					entry.pn = kInfinitePnDn;
					entry.dn = 0;
					entry.num_searched = REPEAT;
				}
			}
			entry.unlock();
			return;
		}
		break;
	}

	const Ply game_ply = n.gamePly();
	bool is_skip = false;
	// 定期的にroot に戻ってきて、また再度自身の担当depth へ現状の最良手順で辿っていく、のを繰り返すはず。。。
	if (threadid && !is_root && game_ply <= threadid) {    // メインスレッドとルートノードはskip しない
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
    //     : 以下の情報があれば、置換表に直接アクセス(LookUpDirect)出来る。
    //     : 以下ではloop 度々子ノードにアクセスするのでキャッシュしておく。
    // 子局面のハッシュエントリをキャッシュ
	struct TTKey {
		TranspositionTable::Cluster* entries;
		uint32_t hash_high;
		Hand hand;
	} ttkeys[MaxCheckMoves];
	for (const auto& move : move_picker) {
		auto& ttkey = ttkeys[&move - move_picker.begin()];

		transposition_table->GetChildFirstEntry<or_node>(n, move, ttkey.entries, ttkey.hash_high, ttkey.hand);

	}

	while (true) {
		//if (threadid == 23) {
		//	std::stringstream ss;
		//	ss << "W";
		//	enqueue_inf_debug_str(ss.str());
		//}

		// NOTE: 恐らくnum_searched について代入でない更新がなされるのはここだけ。
		entry.lock();
		// NOTE
		//     : 並列実行されるのでこうしないといけない。
		if (entry.num_searched != REPEAT) {
			++entry.num_searched;    // TODO: これだけの為にlock するのはもったいないよな。どっか移動したい。。。
		}
//#ifdef DFPN_MOD_V8
		const auto entry_num_searched = entry.num_searched;
//#endif

		// debug
		bool is_debug_root = false;
		if (entry.hash_high == DEBUG_HASH_0 >> 32) {
			is_debug_root = true;
		}

		bool is_debug_ch_root = false;
		if (entry.hash_high == DEBUG_CH_HASH_0 >> 32) {
			is_debug_ch_root = true;
		}

		std::vector<std::string> debug_str;
		debug_str.reserve(move_picker.size());

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
			Hand new_hand;
			// 子局面の反証駒の積集合
			u32 pawn = UINT_MAX;
			u32 lance = UINT_MAX;
			u32 knight = UINT_MAX;
			u32 silver = UINT_MAX;
			u32 gold = UINT_MAX;
			u32 bishop = UINT_MAX;
			u32 rook = UINT_MAX;
			bool repeat = false; // 最大手数チェック用
			//int tmpCount = 0;    // debug
			for (const auto& move : move_picker) {
				auto& ttkey = ttkeys[&move - move_picker.begin()];
				// TODO
				//     : これlock するか、完全なコピーを取ってそいつを使うか、どっちが速いだろうか。テストすべきかな～？
				//       というか、途中で結果変わるの割と不都合な気もするな。知らんけど。
				//       _mtx はコピーしなくて良いから、struct TTEntryNoMtx を作って、そいつにコピー使用かしら。
				//const auto& child_entry = transposition_table->LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				auto& child_entry = transposition_table->LookUpDirect<ReadOnly>(
					*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1
				);    // 本当はconst auto&
				child_entry.lock();

				// debug
				if (is_debug_root) {
					//if (child_entry.hash_high == DEBUG_CHILD_HASH_0 >> 32 && child_entry.pn == kInfinitePnDn) {
					if (child_entry.pn >= kInfinitePnDn) {
						std::stringstream ss;
						ss
							<< "OrRootChildInf:id = " << threadid << ",move=" << move.move.toUSI() << ",ply=" << n.gamePly()
							<< ",pn=" << child_entry.pn << ",dn=" << child_entry.dn << ",depth=" << child_entry.depth
							<< ",hand=" << child_entry.hand.value() << ",num_searched=" << child_entry.num_searched;
						debug_str.emplace_back(ss.str());
					}
				}
				if (is_debug_ch_root) {
					//if (child_entry.hash_high == DEBUG_CHILD_HASH_0 >> 32 && child_entry.pn == kInfinitePnDn) {
					if (child_entry.pn >= kInfinitePnDn) {
						std::stringstream ss;
						ss
							<< "OrRootChildInf[ch]:id = " << threadid << ",move=" << move.move.toUSI() << ",ply=" << n.gamePly()
							<< ",pn=" << child_entry.pn << ",dn=" << child_entry.dn << ",depth=" << child_entry.depth
							<< ",hand=" << child_entry.hand.value() << ",num_searched=" << child_entry.num_searched;
						debug_str.emplace_back(ss.str());
					}
				}

				// NOTE
				//     : ここでは、"depth が異なるが、優越関係を満たしたら返す"、がワンちゃん活きる。
				//       何故なら、LookUp したentry のpn, dn を(真に(pn == 0を詰みとして扱っている))使ってるからである。
				//       -> えーっと、後半日本語崩壊してない？大丈夫？(@20231128)
				if (child_entry.pn == 0) {    // NOTE: OrNode で子ノードの一つが詰みだったので、現在のノードも詰み
					// 詰みの場合

					new_hand = child_entry.hand;
					child_entry.unlock();

					// 子局面の証明駒を設定
                    // 打つ手ならば、証明駒に追加する
					if (move.move.isDrop()) {
						const HandPiece& hp = move.move.handPieceDropped();
						new_hand.plusOne(hp);    // 差分更新
					}
					// 後手の駒を取る手ならば、証明駒から削除する
					else {
						const Piece& to_pc = n.piece(move.move.to());
						if (to_pc != Empty) {
							const PieceType& pt = pieceToPieceType(to_pc);
							const HandPiece& hp = pieceTypeToHandPiece(pt);
							if (new_hand.exists(hp)) {
								new_hand.minusOne(hp);    // 差分更新
							}
						}
					}

					entry_pn = 0;
					entry_dn = kInfinitePnDn;

					// NOTE: ここはOrNode,、つまり詰みについてOr で、詰みが見つかったのでbreak
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
					// NOTE
					//     : child_entry.num_searched != REPEAT のチェックはこの後する。
					//       仮にchild_entry.num_searched == REPEAT なら反証駒のset はしないので、
					//       以下の情報は使用されず、問題ない。
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
				//     : 
				entry_pn = std::min(entry_pn, child_entry.pn);
				entry_dn += child_entry.dn;

				// TODO
				//     : これ、上の箇所でやったら良くないか？ダメなのかね？
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

			entry.lock();

			// NOTE
			//     : OrNode は不詰みに関してAnd なので、全部見てからでないとdn == 0 は途中では言い切れない。
			//       全部見終わったので、ここで初めて"dn == 0 なら不詰みである" が真となる。
			//     : df-pn の鉄則「詰み || 不詰み が判明したら、証明駒 || 反証駒 を計算して置換表にset」
			//     : この今の局面が不詰みである(entry_dn==0)から反証駒をset するのである。
			//       いつ何時勝手に変更されているか分からないentry.dn の値を用いてはならない。
			//std::cout << "update" << std::endl;
			if (entry_dn == 0) {

				for (const auto& outStr : debug_str) {
					// debug
					sync_cout
						<< "OrRootChildInf:move_picker.size()=" << move_picker.size() << sync_endl;
					sync_cout << outStr << sync_endl;
				}

				if (entry.pn != 0) {
					if (repeat) {
						//std::cout << "repeat, " << entry.pn << "," << entry.dn << "," << entry.depth << std::endl;
						// NOTE
						//     : 最大手数で引き分けなら、初めてentry.dn = 0 とする時のみ 最大手数引き分けを登録する。
						//       初めてでない場合、
						//       (a) 現在、既に登録されている情報が最大手数引き分け不詰み
						//               汎用性は同じなので更新する必要はない。
						//       (b) 現在、既に登録されている情報が、非最大手数引き分け不詰み
						//               既に登録されている情報の方が、反証駒が利用出来るので汎用性が高いので、更新の必要ないし、してはならない。
						//       以上より、更新する必要ない。
						if (entry.dn != 0) {
							// debug
							if (is_debug_root) {
								sync_cout
									<< "debug: pn=inf OrNode com (rep)" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
									<< ",depth=" << entry.depth << ",hand=" << entry.hand.value()
									<< ",new_hand=" << new_hand.value() << ",pos_hand=" << n.hand(n.turn()).value() << sync_endl;
							}
							if (is_debug_ch_root) {
								sync_cout
									<< "debug[ch]: pn=inf OrNode com (rep)" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
									<< ",depth=" << entry.depth << ",hand=" << entry.hand.value()
									<< ",new_hand=" << new_hand.value() << ",pos_hand=" << n.hand(n.turn()).value() << sync_endl;
							}

							entry.pn = kInfinitePnDn;
							entry.dn = 0;
							entry.num_searched = REPEAT;
						}
					}
					else {

						// debug
						if (is_debug_root) {
							sync_cout
								<< "debug: pn=inf OrNode com (normal)" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value()
								<< ",new_hand=" << new_hand.value() << ",pos_hand=" << n.hand(n.turn()).value() << sync_endl;
						}
						if (is_debug_ch_root) {
							sync_cout
								<< "debug[ch]: pn=inf OrNode com (normal)" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value()
								<< ",new_hand=" << new_hand.value() << ",pos_hand=" << n.hand(n.turn()).value() << sync_endl;
						}

						// NOTE
						//     : single thread では、一度pn == 0 or dn == 0 となった局面に訪問することはないが、
						//       multi nthread の場合は訪問する局面がスレッド同士で被ってしまう事がある。
						//       その場合は、entry.hand を先手の持ち駒として処理するのは間違いとなるので、
						//       position から確実に先手の持ち駒を貰う。
						// 先手が一枚も持っていない種類の先手の持ち駒を反証駒から削除する
						new_hand.set(0);
						const Hand& hand_attacker = n.hand(n.turn());    // 攻め側の持ち駒
						u32 curr_pawn   = hand_attacker.template exists<HPawn>();   if (curr_pawn == 0)   pawn = 0;   else if (pawn < curr_pawn)     pawn = curr_pawn;
						u32 curr_lance  = hand_attacker.template exists<HLance>();  if (curr_lance == 0)  lance = 0;  else if (lance < curr_lance)   lance = curr_lance;
						u32 curr_knight = hand_attacker.template exists<HKnight>(); if (curr_knight == 0) knight = 0; else if (knight < curr_knight) knight = curr_knight;
						u32 curr_silver = hand_attacker.template exists<HSilver>(); if (curr_silver == 0) silver = 0; else if (silver < curr_silver) silver = curr_silver;
						u32 curr_gold   = hand_attacker.template exists<HGold>();   if (curr_gold == 0)   gold = 0;   else if (gold < curr_gold)     gold = curr_gold;
						u32 curr_bishop = hand_attacker.template exists<HBishop>(); if (curr_bishop == 0) bishop = 0; else if (bishop < curr_bishop) bishop = curr_bishop;
						u32 curr_rook   = hand_attacker.template exists<HRook>();   if (curr_rook == 0)   rook = 0;   else if (rook < curr_rook)     rook = curr_rook;

						// 反証駒に子局面の証明駒の積集合を設定
						new_hand.set(pawn | lance | knight | silver | gold | bishop | rook);

						// NOTE
						//     : 反証駒どうするよ？
						//       -> 先手の任意の指し手と、それによって遷移した先のある子ノードのエントリのhand から差分計算によって
						//          求めたhand の積集合をとれば良い。
						//          ただ、Komoring heights のblog 曰く、さらにB_n と積集合を取らないといけないらしいが、何故なのか分からん。
						//          後、AndNode での証明駒の処理も一応見ておいて、何か抜けが無いか確認する。
						//       -> あれあらしい、後手が独占している持ち駒を、反証駒から削除らしい。
						//          だから、上の実装があればそれでok
						UpdateDnZero(entry, kInfinitePnDn, new_hand);
						//std::cout << "done: " << entry.pn << "," << entry.dn << "," << entry.depth << std::endl;
					}

				}
			}
			else {
				if (entry_pn == 0) {
					if (entry.dn != 0) {
						UpdatePnZero(entry, kInfinitePnDn, new_hand);
						//std::cout << "done: " << entry.pn << "," << entry.dn << "," << entry.depth << std::endl;
					}
				}
				else {
					if (entry.pn != 0 && entry.dn != 0) {    // 書き込む値の両方が完全に未知の場合、元のpn, dn どちらもチェックする。
		        		// TODO20231026: 上書きのみ
						// debug
						if (is_debug_root && entry_pn >= kInfinitePnDn) {
							sync_cout
								<< "debug: pn=inf OrNode com" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value() << ",entry_pn=" << entry_pn << ",entry_dn=" << entry_dn 
								<< ",new_hand=" << new_hand.value() << ",pos_hand=" << n.hand(n.turn()).value() << sync_endl;
						}
						if (is_debug_ch_root && entry_pn >= kInfinitePnDn) {
							sync_cout
								<< "debug[ch]: pn=inf OrNode com" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value() << ",entry_pn=" << entry_pn << ",entry_dn=" << entry_dn
								<< ",new_hand=" << new_hand.value() << ",pos_hand=" << n.hand(n.turn()).value() << sync_endl;
						}

						//entry.pn = entry_pn;
						entry.pn = (entry_pn >= kInfinitePnDn ? kInfinitePnDn : entry_pn);    // debug用に一時的にこちらに変更
		        		entry.dn = std::min(entry_dn, kInfinitePnDn);
						//std::cout << "done: " << entry.pn << "," << entry.dn << "," << entry.depth << std::endl;
					}
				}

				// NOTE
				//     : if (entry.pn >= thpn || entry.dn >= thdn) { break; } と評価しており、
				//       第二候補の兄弟局面のスコアを超過して初めて打ち切りたいので、"+1" する。
				if constexpr (EPS == 0) {    // 標準
					thpn_child = std::min({ thpn, second_best_pn + 1, kInfinitePnDn });
				}
				else if constexpr (EPS > 0) {
#ifdef DFPN_MOD_V9
					if (entry_num_searched < EPS_MULTIPLY_NS_THRESHOLD) {
						float ns_rest = EPS_MULTIPLY_NS_THRESHOLD - entry_num_searched;
						min_clip<float>(ns_rest, EPS_MULTIPLY_LOG_BASE);

						float adaptive_multiply = logbase<float>(ns_rest, EPS_MULTIPLY_LOG_BASE);
						max_clip<float>(adaptive_multiply, _EPS_PLUS_ONE);

						thpn_child = std::min({ thpn, static_cast<int>(adaptive_multiply * second_best_pn + 1), kInfinitePnDn });
					}
					else {
						thpn_child = std::min({ thpn, static_cast<int>(_EPS_PLUS_ONE * second_best_pn + 1), kInfinitePnDn });
					}
#else
					// より子ノードの長く滞在
					thpn_child = std::min({ thpn, static_cast<int>(_EPS_PLUS_ONE * second_best_pn + 1), kInfinitePnDn });
#endif

					//if (game_ply == threadid) {
					//	// より子ノードの長く滞在
					//	thpn_child = std::min(thpn, static_cast<int>(_EPS_PLUS_ONE * second_best_pn + 1));
					//}
					//else {
					//	thpn_child = std::min(thpn, second_best_pn + 1);
					//}
				}
				else {
					// より子ノードに短く滞在
					thpn_child = std::min(
						{ thpn, best_pn + static_cast<int>(_EPS_PLUS_ONE * (second_best_pn - best_pn) + 1), kInfinitePnDn }
					);

					//if (game_ply == threadid) {
					//	// より子ノードに短く滞在
					//	thpn_child = std::min(
					//		thpn, best_pn + static_cast<int>(_EPS_PLUS_ONE * (second_best_pn - best_pn) + 1)
					//	);
					//}
					//else {
					//	thpn_child = std::min(thpn, second_best_pn + 1);
					//}
				}
				thdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn);

				//// debug
				//if (thdn_child < 0 && threadid == 0) {
				//	sync_cout << "[OrDn:thdn_child=" << thdn_child << ",thdn=" << thdn << ",entry.dn=" << entry.dn << ",best_pn=" << best_pn << ",best_dn=" << best_dn << "]" << sync_endl;
				//}
				//if (thpn_child < 0 && threadid == 0) {
				//	sync_cout << "[OrPn:thpn_child=" << thpn_child << ",thpn=" << thpn << ",second_best_pn=" << second_best_pn << "]" << sync_endl;
				//}
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
			int entry_dn = kInfinitePnDn;    // TODO: これ、inf+3 にした方が良いかね？それともせんでも良い？後で考える。
#else
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
#endif

			Hand new_hand;
			// 子局面の証明駒の和集合
			u32 pawn = 0;
			u32 lance = 0;
			u32 knight = 0;
			u32 silver = 0;
			u32 gold = 0;
			u32 bishop = 0;
			u32 rook = 0;
			bool repeat = false;
			bool all_mate = true;    // NOTE: これまでに処理した子ノードの全てが詰みである限りtrue
			for (const auto& move : move_picker) {
				auto& ttkey = ttkeys[&move - move_picker.begin()];
				//const auto& child_entry = transposition_table->LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				// TODO20231026: 情報の参照のみ。
				auto& child_entry = transposition_table->LookUpDirect<ReadOnly>(
					*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1
				);    // 本当はconst auto&
				child_entry.lock();

				// debug
				if (is_debug_root) {
					if (child_entry.pn >= kInfinitePnDn) {
						sync_cout
							<< "AndRootChild:id = " << threadid << ",move=" << move.move.toUSI()
							<< ",pn=" << child_entry.pn << ",dn=" << child_entry.dn << ",depth=" << child_entry.depth
							<< ",hand=" << child_entry.hand.value() << ",num_searched=" << child_entry.num_searched << sync_endl;
					}
				}
				if (is_debug_ch_root) {
					if (child_entry.pn >= kInfinitePnDn) {
						sync_cout
							<< "AndRootChild[ch]:id = " << threadid << ",move=" << move.move.toUSI()
							<< ",pn=" << child_entry.pn << ",dn=" << child_entry.dn << ",depth=" << child_entry.depth
							<< ",hand=" << child_entry.hand.value() << ",num_searched=" << child_entry.num_searched << sync_endl;
					}
				}

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
					
					// 最大手数で不詰みの局面が優越関係で使用されないようにする
					if (child_entry.num_searched == REPEAT) {
						child_entry.unlock();

						repeat = true;
						// 反証駒は計算しないし出来ない(難しい?)。
					}
					else {
						new_hand = child_entry.hand;
						child_entry.unlock();
#ifdef DFPN_MOD_V7
						if (move.move.isDrop()) {
							const HandPiece& hp = move.move.handPieceDropped();
							const int& dp_hp_cnt = new_hand.numOf(hp);
							const int& actual_total_hp_cnt = n.hand(n.turn()).numOf(hp) + n.hand(oppositeColor(n.turn())).numOf(hp);
							// 反証駒にある駒hp が、実際の先手の持ち駒と同じかそれ以上持っていた場合、move を打てない。
							if (actual_total_hp_cnt <= dp_hp_cnt) {
								// 後手が持ち駒から打てるように、actual_total_hp_cnt - 1個 にset する。
								new_hand.minusN(hp, dp_hp_cnt);
								new_hand.plusN(hp, actual_total_hp_cnt - 1);
							}
						}
#else
						// 子局面の反証駒を設定
						// 打つ手ならば、反証駒から削除する
						if (move.move.isDrop()) {
							const HandPiece& hp = move.move.handPieceDropped();
							if (new_hand.exists(hp)) {
								new_hand.minusOne(hp);
							}
						}
						// 先手の駒を取る手ならば、反証駒に追加する
						else {
							const Piece& to_pc = n.piece(move.move.to());
							if (to_pc != Empty) {
								const PieceType& pt = pieceToPieceType(to_pc);
								const HandPiece& hp = pieceTypeToHandPiece(pt);
								new_hand.plusOne(hp);
							}
						}
#endif
					}
					
					entry_pn = kInfinitePnDn;
					entry_dn = 0;

					// NOTE: ここはAndNode,、つまり不詰みについてOr で、不詰みが見つかったのでbreak
					break;
				}

				// NOTE
				//     : 
				entry_pn += child_entry.pn;
				entry_dn = std::min(entry_dn, child_entry.dn);

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
				child_entry.unlock();
			}    // NOTE: for 終了
			entry.lock();

			// NOTE
			//     : ここで初めて"pn == 0 ならば詰みである" が真となる。
			//     : df-pn の鉄則より、反証駒の計算とset
			//     : UpdateOnly では同じdepth しか読まないので、本当は詰みの局面で、別の手数の最大手数不詰みの局面の結果を参照してしまうことは無いけど、
			//       子ノードはReadOnly 故、手数違いの局面の情報も参照する。
			//       なので、UpdateOnly で読んでも、更新するときに異なるdepth の情報によって更新されることで、
			//       子ノードの手数違いの最大手数不詰み局面の情報が混じってしまう。
			//       -> あー、これじゃ不正確だな。正確に言おう。
			//          LookUpDirect では、異なるdepth の局面を返す時はnum_searched != REPEAT の時のみ返している。
			//          なので、LookUpDirect から間違った(用いてはならない情報) が返されることは無いはずだ。
			//          だが、compute pn and dn ではどうだろうか？
			//          get_pv_inner でも起きたように、先手が手数を考えず、兎に角詰む手を一つ選んでいくと、
			//          恐らく、どこかで、先手が手数最小化の努力もすれば制限深さ以内に詰むはずが、
			//          舐めプの指し手を指したせいで最大手数不詰みとなり、不当に最大手数不詰みが登録される気がする。
			//          (まだ未検証だけど、これが一番有力だと勝手に思い込んでいる。)
			//     : 軽めの対処として、dn == 0 && num_searched == REPEAT なら結果を覆して良い、みたいなのはありだと思う。
			if (entry_pn == 0) {
				if (entry.dn != 0) {
					// NOTE: 証明駒が、現在の先手の持ち駒より大きくなることはあり得ない。(だって現在の先手の持ち駒で詰んでるから。)
					new_hand.set(0);
					const Hand& hand_attacker = n.hand(oppositeColor(n.turn()));    // 攻め側の持ち駒
					u32 curr_pawn   = hand_attacker.template exists<HPawn>();   if (pawn > curr_pawn)     pawn = curr_pawn;
					u32 curr_lance  = hand_attacker.template exists<HLance>();  if (lance > curr_lance)   lance = curr_lance;
					u32 curr_knight = hand_attacker.template exists<HKnight>(); if (knight > curr_knight) knight = curr_knight;
					u32 curr_silver = hand_attacker.template exists<HSilver>(); if (silver > curr_silver) silver = curr_silver;
					u32 curr_gold   = hand_attacker.template exists<HGold>();   if (gold > curr_gold)     gold = curr_gold;
					u32 curr_bishop = hand_attacker.template exists<HBishop>(); if (bishop > curr_bishop) bishop = curr_bishop;
					u32 curr_rook   = hand_attacker.template exists<HRook>();   if (rook > curr_rook)     rook = curr_rook;

					// 証明駒に子局面の証明駒の和集合を設定
					new_hand.set(pawn | lance | knight | silver | gold | bishop | rook);

				    // NOTE
				    //     : 進研ゼミでやったところ
				    //     : ここはAndNode なので、setPP() するときは、受け側が一枚も持ってなくて、攻め手が独占してる駒を
				    //       証明駒に追加しないといけない。つまり、them にoppositeColor のhand を渡すべきでは？
				    //     : 条件の解説
				    //         : n.checkersBB() & n.attacksFrom<King>(n.kingSquare(n.turn())) は、
				    //           後手に王手を掛けてる駒のBB & "後手玉の位置"に位置する玉(つまり後手玉)の利きのBB
				    //           -> 近接王手をしてる駒が一つでもあればtrue
				    //         : n.checkersBB() & n.attacksFrom<Knight>(n.turn(), n.kingSquare(n.turn())))
				    //           後手に王手を掛けてる駒のBB & "後手玉の位置"に位置する後手番の桂馬の利きのBB
				    //           -> 後手に桂馬で王手を掛けれる位置に、後手玉に王手を掛けてる駒が一つでもあればtrue
				    //       -> (近接王手が一つもない && 桂馬王手が一つもない)
				    //       -> (合駒で防ぐことが絶対に出来ない王手(近接王手or桂馬王手) が一つもない)
				    //       -> (遠隔王手(∵詰みの局面なので何かしらの王手は掛かってる。
					//           つまり、上記の種類の王手の補集合の王手が掛かってるはず。))
				    // 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
					if (!(n.checkersBB() & n.attacksFrom<King>(n.kingSquare(n.turn()))
						|| n.checkersBB() & n.attacksFrom<Knight>(n.turn(), n.kingSquare(n.turn())))
						) {
						new_hand.setPP(n.hand(oppositeColor(n.turn())), n.hand(n.turn()));
					}

					UpdatePnZero(entry, kInfinitePnDn, new_hand);
				}
			}
			else {
				if (entry_dn == 0) {
					if (entry.pn != 0) {
						if (repeat) {
							if (entry.dn != 0) {
								// debug
								if (is_debug_root) {
									sync_cout
										<< "debug: pn=inf at AndNode compute pn dn (rep)" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
										<< ",depth=" << entry.depth << ",hand=" << entry.hand.value() << sync_endl;
								}
								if (is_debug_ch_root) {
									sync_cout
										<< "debug[ch]: pn=inf at AndNode compute pn dn (rep)" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
										<< ",depth=" << entry.depth << ",hand=" << entry.hand.value() << sync_endl;
								}

								entry.pn = kInfinitePnDn;
								entry.dn = 0;
								entry.num_searched = REPEAT;
							}
						}
						else {
						// TODO20231103
						//     : 反証駒どうするよ？
						//       -> 後手がある指し手 を指したことで詰みを逃れて不詰みになったんだから、
						//          new_hand = そのある指し手 による遷移先の反証駒; じゃない？
						//          後手がそのある指し手 を指すことによる反証駒の変化は無いはず。(∵後手の任意の指し手は先手の持ち駒に影響を与えない。)
							UpdateDnZero(entry, kInfinitePnDn, new_hand);
						}
					}
				}
				else {
					if (entry.pn != 0 && entry.dn != 0) {    // 書き込む値の両方が完全に未知の場合、元のpn, dn どちらもチェックする。
						// debug
						if (is_debug_root && entry_pn >= kInfinitePnDn) {
							sync_cout
								<< "debug: pn=inf at AndNode compute pn dn" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value() << ",entry_pn=" << entry_pn << ",entry_dn=" << entry_dn << sync_endl;
						}
						if (is_debug_ch_root && entry_pn >= kInfinitePnDn) {
							sync_cout
								<< "debug[ch]: pn=inf at AndNode compute pn dn" << ",id=" << threadid << ",pn=" << entry.pn << ",dn=" << entry.dn
								<< ",depth=" << entry.depth << ",hand=" << entry.hand.value() << ",entry_pn=" << entry_pn << ",entry_dn=" << entry_dn << sync_endl;
						}

						//entry.pn = std::min(entry_pn, kInfinitePnDn);
						entry.pn = (entry_pn >= kInfinitePnDn ? kInfinitePnDn : entry_pn);    // debug用に一時的にこちらに変更
						entry.dn = entry_dn;
					}
				}

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
					thdn_child = std::min({ thdn, second_best_dn + 1, kInfinitePnDn });
				}
				else if constexpr (EPS > 0) {

#ifdef DFPN_MOD_V9
					if (entry_num_searched < EPS_MULTIPLY_NS_THRESHOLD) {
						float ns_rest = EPS_MULTIPLY_NS_THRESHOLD - entry_num_searched;
						min_clip<float>(ns_rest, EPS_MULTIPLY_LOG_BASE);

						float adaptive_multiply = logbase<float>(ns_rest, EPS_MULTIPLY_LOG_BASE);
						max_clip<float>(adaptive_multiply, _EPS_PLUS_ONE);

						thdn_child = std::min({ thdn, static_cast<int>(adaptive_multiply * second_best_dn + 1), kInfinitePnDn });
					}
					else {
						thdn_child = std::min({ thdn, static_cast<int>(_EPS_PLUS_ONE * second_best_dn + 1), kInfinitePnDn });
					}
#else

					// より子ノードの長く滞在
					thdn_child = std::min({ thdn, static_cast<int>(_EPS_PLUS_ONE * second_best_dn + 1), kInfinitePnDn });
#endif
					//// NOTE: skip しなくなる直前のみに。
					//if (game_ply == threadid) {
					//	// より子ノードの長く滞在
					//	thdn_child = std::min(thdn, static_cast<int>(_EPS_PLUS_ONE * second_best_dn + 1));
					//}
					//else {
					//	thdn_child = std::min(thdn, second_best_dn + 1);
					//}
				}
				else {
					// より子ノードに短く滞在
					thdn_child = std::min(
						{ thdn, best_dn + static_cast<int>(_EPS_PLUS_ONE * (second_best_dn - best_dn) + 1), kInfinitePnDn }
					);

					//if (game_ply == threadid) {
					//	// より子ノードに短く滞在
					//	thdn_child = std::min(
					//		thdn, best_dn + static_cast<int>(_EPS_PLUS_ONE * (second_best_dn - best_dn) + 1)
					//	);
					//}
					//else {
					//	thdn_child = std::min(thdn, second_best_dn + 1);
					//}
				}

				//// debug
				//if (thpn_child < 0 && threadid == 0) {
				//	sync_cout << "[AndPn:" << thpn_child << "," << thpn << "," << entry.pn << "," << best_pn << "]" << sync_endl;
				//}
				//if (thdn_child < 0 && threadid == 0) {
				//	sync_cout << "[AndDn:" << thdn_child << "," << thdn << "," << second_best_dn << "]" << sync_endl;
				//}
			}
			//entry.unlock();    // NOTE: 閾値確認の終わりに移動
		}
		// NOTE
		//     : 詰み(pn == 0) と分かった場合は、dn == inf となっているので、dn の制限でbreak する。(dn == 0 の場合も同様)
		//     : シングルスレッドの場合、pn == 0 || dn == 0 と分かった局面に再度訪問することはない。
		//       (証明)
		//       ある子ノードがpn == 0 を満たす場合を考える。
		//       但し、expand and compute pn(n) and dn(n) において、親ノードがpn == 0 || dn == 0 に更新された場合、
		//       すぐ下の閾値判定によってbreak; するため、親ノードの親ノードへと帰っていき、再帰しないことに注意する。
		//       (1) ある AndNode nにおいて、ある指し手M_0 が存在して、遷移先の子ノードnode(n, M_0)がpn == 0, dn == inf を満たしたとする。
		//               (a) 指し手がM_0 一つだけしか存在しない場合
		//                       この時、expand and compute pn(n) and dn(n) によって親ノードn もpn == 0 となる。
		//               (b) 指し手が二つ以上存在する場合
		//                       (a-1) ある指し手M_jが存在して、遷移先の子ノードがpn != 0, dn < inf = node(n, M_0).dn を満たす。
		//                                 この時、よりdn の少ない指し手M_j が存在するため、指し手M_0 が訪問先として選択されることは無い。
		//                       (b-2) 任意の指し手の、遷移先の子ノードがpn == 0, dn == inf を満たす。
		//                                 この時、expand and compute pn(n) and dn(n) によって親ノードn もpn == 0 となる。
		//       (2) ある OrNode nにおいて、ある指し手M_0 が存在して、遷移先の子ノードnode(n, M_0)がpn == 0, dn == inf を満たしたとする。
		//               この時、expand and compute pn(n) and dn(n) によって親ノードn もpn == 0 となる。
		//       以上より、ある子ノードがpn == 0 を満たす場合、その子ノードには遷移しないか、そもそも遷移(再帰)自体が起きないかのどちらかであり、
		//       pn == 0 と分かった子ノードに再度訪問しないことが証明された。dn == 0の子ノードについても同様の方法で示せる。
		// if (pn(n) >= thpn || dn(n) >= thdn) break; // termination condition is satisfied
		if (entry.pn >= thpn || entry.dn >= thdn || node_count_down == 0) {
#ifdef DEBUG_PRINT_WHEN_ROOT_BREAK_20231101_0
			// debug
			if (entry.depth == 1) {
				sync_cout
					<< "break,id = " << threadid << ",ply=" << n.gamePly()
					<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth << ",hand=" << entry.hand.value()
					<< ",thpn = " << thpn << ",thdn = " << thdn << sync_endl;
			}
#endif

			entry.unlock();
			break;
		}

		// 終了するなら、現在のノードへの伝搬が完了してから抜ける。
		if (searchedNode >= maxSearchNode || _should_stop<shared_stop>()) {
			entry.unlock();
			break;
		}



#ifdef DEBUG_PRINT_ROOT_INFO_20231103
		if (n.gamePly() == 1 &&  threadid == 0) {
			sync_cout
				<< "root,id = " << threadid << ",ply=" << n.gamePly()
				<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth << ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched
				<< ",thpn = " << thpn << ",thdn = " << thdn << sync_endl;
		}
#endif
		//// debug
		//if ((thpn_child < 0 || thdn_child < 0) && threadid == 0) {
		//	sync_cout
		//		<< "[NotBreak:thpn_child=" << thpn_child << ",thdn_child=" << thdn_child
		//		<< ",thpn=" << thpn << ",thdn=" << thdn << ",entry.pn=" << entry.pn << ",entry.dn=" << entry.dn << "]" << sync_endl;
		//}
		//if (best_move.value() == 0) {
		//	sync_cout << "[best_move==0:id=" << threadid << "]" << std::flush << sync_endl;
		//	if (or_node) {
		//		sync_cout << "[Or:id=" << threadid << ",move_picker.size()=" << move_picker.size()
		//			<< ",ply=" << n.gamePly() << ",depth=" << entry.depth
		//			<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",thpn=" << thpn << ",thdn=" << thdn << "]" << std::flush << sync_endl;
		//	}
		//	else {
		//		sync_cout << "Andr:id=" << threadid << ",move_picker.size()=" << move_picker.size()
		//			<< ",ply=" << n.gamePly() << ",depth=" << entry.depth
		//			<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",thpn=" << thpn << ",thdn=" << thdn << "]" << std::flush << sync_endl;
		//	}
		//}

		// debug
		const int entry_depth = entry.depth;

		entry.unlock();

		// skip しなくなる直前でのみ実行
		//if (game_ply <= threadid && entry_num_searched < TH_MULTIPLY_NS_THRESHOLD) {    // game_ply == threadid より若干速くなったが、これはmultiply 自体を大きくすればそれと同等になるのでは？
		//if (game_ply >= threadid && entry_num_searched < TH_MULTIPLY_NS_THRESHOLD) {    // ある局面で探索が進まなくなった。
		//if (entry_num_searched < TH_MULTIPLY_NS_THRESHOLD) {     // ある局面で探索が進まなくなった。
		if (is_skip && game_ply == threadid && entry_num_searched < TH_CH_MULTIPLY_NS_THRESHOLD) {    // is_skip は無駄な評価。
		//if (is_skip && game_ply == threadid) {
		//if (is_skip) {
		//if (true) {    // これだと終わらなくなった。
#ifdef DFPN_MOD_V8
			// 探索回数が少ないほどadaptive_multiply は大きいが、TH_CH_MULTIPLY は超えない。
			// 探索回数がTH_CH_MULTIPLY_NS_THRESHOLD に近づくほど小さくして1に近づける。
			float ns_rest = TH_CH_MULTIPLY_NS_THRESHOLD - entry_num_searched;
			min_clip<float>(ns_rest, TH_MULTIPLY_LOG_BASE);
			
			float adaptive_multiply = logbase<float>(ns_rest, TH_MULTIPLY_LOG_BASE);
			max_clip<float>(adaptive_multiply, TH_CH_MULTIPLY);

			thpn_child = std::min(static_cast<int>(thpn_child * adaptive_multiply), kInfinitePnDn);
			thdn_child = std::min(static_cast<int>(thdn_child * adaptive_multiply), kInfinitePnDn);
#else
			// TODO
			//     : 深さに応じてこれ変えるとか、そもそもを3とか4にするとかね？色々ある。
			thpn_child = std::min(static_cast<int>(thpn_child * TH_CH_MULTIPLY), kInfinitePnDn);
			thdn_child = std::min(static_cast<int>(thdn_child * TH_CH_MULTIPLY), kInfinitePnDn);
#endif
		}

		volatile const std::string best_move_usi = best_move.toUSI();

		StateInfo state_info;
		n.doMove(best_move, state_info);
		++searchedNode;

		//// debug
		if (entry_depth % 2 == n.gamePly() % 2) {
			entry.lock();
			sync_cout
				<< "[plyDepthMismatch:id=" << threadid << ",best_move=" << best_move.value() << ",thpn_child=" << thpn_child << ",thdn_child=" << thdn_child
				<< ",thpn=" << thpn << ",thdn=" << thdn << ",entry.pn=" << entry.pn << ",entry.dn=" << entry.dn << "]" << std::flush << sync_endl;
			entry.unlock();
		}
		//if ((thpn_child < 0 || thdn_child < 0)) {
		//	sync_cout
		//		<< "[NotBreak:id=" << threadid << ",thpn_child=" << thpn_child << ",thdn_child=" << thdn_child
		//		<< ",thpn=" << thpn << ",thdn=" << thdn << ",entry.pn=" << entry.pn << ",entry.dn=" << entry.dn << "]" << sync_endl;
		//}

		// 次のノードがroot であることはあり得ない。
		dfpn_inner<!or_node, shared_stop, false>(n, thpn_child, thdn_child/*, inc_flag*/, maxDepth, searchedNode, threadid);
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
		const auto& child_entry = transposition_table->LookUpChildEntry<true, ReadOnly>(pos, move);
		if (child_entry.pn == 0) {
			return move;
		}
	}

	return Move::moveNone();
}

// FIXME: 永遠に循環して終わらないことがあるっぽい。。。？
template<bool or_node, bool safe>
int ParallelDfPn::get_pv_inner(Position& pos, std::vector<Move>& pv) {
	// debug
	std::ostringstream oss;
	pos.print(oss);
	const auto& root_key = pos.getBoardKey();

	// NOTE
	//     : Or は最善の手順を探索せずに、ひとまず詰めば良し、としている。
	//       -> これ、双方最善尽くさないとダメやん？
	//          だって、頭金で詰みなのに、指し手引き延ばす指し手(つまり舐めプの手)の方が先に来たら、その指し手を選択するんやろ？
	//          じゃあ、後手はその手順の方が寿命が長いと勘違いするよね？
	//          すると、読み筋の後手の着手も必然的に最善とは限らなくなる。
	//       -> いや、まぁ、でもそんなもんか。
	if (or_node) {
		//std::cout << "[Or][ply = " << pos.gamePly() << "][" << pos.getBoardKey() << "]\n" << ss.str() << std::endl;
		// ORノードで詰みが見つかったらその手を選ぶ
		MovePicker<true> move_picker(pos);

		// debug
		if (root_key == DEBUG_HASH_1) {
			std::cout << "infoOr: is debug tag start"
				<< ",ply=" << pos.gamePly() << ",move_picker.size()=" << move_picker.size() << ",sfen=" << pos.toSFEN() << std::endl;
			std::cout << oss.str() << std::endl;
		}

		// 1手詰め, 3手詰め だけ先に優先して確認。
		// (読み筋上での先手の着手の精度が比較的低コストで若干向上して、変な長い筋にはまりづらくなり、局面によっては超高速化するっぽい。)
		for (const auto& move : move_picker) {
			// NOTE: コイツはlock, unlock 以外はconst
			auto& child_entry = transposition_table->LookUpChildEntry<true, ReadOnly>(pos, move);
			if (safe) child_entry.lock();

			// debug
			if (root_key == DEBUG_HASH_1) {
				std::cout
					<< "infoOr: child_entry" << ",move=" << move.move.value() << ",moveUSI=" << move.move.toUSI() << ",ply=" << pos.gamePly()
					<< ",pn=" << child_entry.pn << ",dn=" << child_entry.dn << ",depth=" << child_entry.depth
					<< ",hand=" << child_entry.hand.value() << ",num_searched=" << child_entry.num_searched << std::endl;

				//if (pos.gamePly() == 17) {
				//	if (move.move.value() == 68115) continue;
				//}
			}

			if (child_entry.pn == 0) {
				if (child_entry.dn == kInfinitePnDn + 1) {
					if (safe) child_entry.unlock();
					pv.emplace_back(move);
					return 1;
				}
				if (child_entry.dn == kInfinitePnDn + 3) {
					if (safe) child_entry.unlock();
					StateInfo si;
					pos.doMove(move, si);

					pv.emplace_back(move);
					const auto depth = get_pv_inner<false, safe>(pos, pv);

					pos.undoMove(move);
					return depth + 1;    // 一応3 になるはずだが、置換表の状態によってはならない事も。
				}
			}
			if (safe) child_entry.unlock();
		}

		//// debug
		//if (root_key == DEBUG_HASH_1) {
		//	std::cout << "infoOr: is debug tag re" << std::endl;
		//}

		for (const auto& move : move_picker) {
			// NOTE: コイツはlock, unlock 以外はconst
			auto& child_entry = transposition_table->LookUpChildEntry<true, ReadOnly>(pos, move);
			if (safe) child_entry.lock();

			//// debug
			//if (root_key == DEBUG_HASH_1) {
			//	std::cout
			//		<< "infoOr: child_entry" << ",move=" << move.move.value() << ",moveUSI=" << move.move.toUSI() << ",ply=" << pos.gamePly()
			//		<< ",pn=" << child_entry.pn << ",dn=" << child_entry.dn << ",depth=" << child_entry.depth << ",hand=" << child_entry.hand.value() << std::endl;

			//	//if (pos.gamePly() == 17) {
			//	//	if (move.move.value() == 68115) continue;
			//	//}
			//}

			//std::cout
			//	<< "[Or][ply = " << pos.gamePly() << "][move_picker] " << Move(move).toUSI()
			//	<< " = [" << child_entry.pn << ", " << child_entry.dn << ", " << child_entry.hand.value() << "]" << std::endl;
			if (child_entry.pn == 0) {
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

		// 実際子ノードに来てみると、手数が違うくて、Repetition だった、みたいな時はここにくる。
		//std::cout
		//	<< "infoOr: no best move" << ",ply=" << pos.gamePly() << ",key=" << root_key << ",sfen=" << pos.toSFEN() << std::endl;
		//std::cout << oss.str() << std::endl;

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
		int max_depth_num_searched = 0;    // max_depth の指し手によって遷移した子ノードのnum_searched
		std::vector<Move> max_pv;    // NOTE: 最大のPV
		MovePicker<false> move_picker(pos);


		//// debug
		//if (root_key == DEBUG_HASH_1) {
		//	std::cout << "infoAnd: is debug tag start"
		//		<< ",ply=" << pos.gamePly() << ",move_picker.size()=" << move_picker.size() << ",sfen=" << pos.toSFEN() << std::endl;
		//	std::cout << oss.str() << std::endl;
		//}

		for (const auto& move : move_picker) {
			// NOTE: コイツはlock, unlock 以外はconst
			auto& child_entry = transposition_table->LookUpChildEntry<false, ReadOnly>(pos, move);
			if (safe) child_entry.lock();
			const auto child_entry_num_searched = child_entry.num_searched;

			//// debug
			//if (root_key == DEBUG_HASH_1) {
			//	std::cout
			//		<< "infoAnd: child_entry" << ",move=" << move.move.value() << ",moveUSI=" << move.move.toUSI() << ",ply=" << pos.gamePly()
			//		<< ",pn=" << child_entry.pn << ",dn=" << child_entry.dn << ",depth=" << child_entry.depth
			//		<< ",hand=" << child_entry.hand.value() << ",num_searched=" << child_entry.num_searched << std::endl;

			//	//if (pos.gamePly() == 17) {
			//	//	if (move.move.value() == 68115) continue;
			//	//}
			//}
			//// debug
			//if (child_entry.hash_high == DEBUG_HASH_1 >> 32) {
			//	std::cout
			//		<< "infoAnd: child_entry" << ",move=" << move.move.toUSI() << ",ply=" << pos.gamePly()
			//		<< ",pn=" << child_entry.pn << ",dn=" << child_entry.dn << ",depth=" << child_entry.depth << ",hand=" << child_entry.hand.value() << std::endl;
			//	std::cout << oss.str() << std::endl;
			//}

			if (child_entry.pn == 0) {
				std::vector<Move> tmp_pv{ move };
				StateInfo state_info;
				pos.doMove(move, state_info);
				int depth = -kInfinitePnDn;
				if (child_entry.dn == kInfinitePnDn + 2) {
					if (safe) child_entry.unlock();

					//std::cout << "[And][child_entry.dn == kInfinitePnDn + 2]" << Move(move).toUSI() << std::endl;
					depth = 1;
					if (!pos.inCheck()) {    // NOTE: mateMoveIn1Ply() が使える条件。
						// 1手詰みチェック
						Move&& mate1ply = pos.mateMoveIn1Ply<mmin1ply_Additional>();
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
					//// debug
					//if (root_key == DEBUG_HASH_1) {
					//	std::cout
					//		<< "infoAnd: child_entry" << ",move=" << move.move.value() << ",moveUSI=" << move.move.toUSI() << ",ply=" << pos.gamePly()
					//		<< ",returnDepth=" << depth << std::endl;
					//}
				}
				pos.undoMove(move);

				// NOTE
				//     : より長いPVが見つかれば更新
				//     : 同じdepth でも、本来はもっと少ないけど大きく見えてるパターンがあり得る。
				//       その場合、num_searched が大きいほどその局面を詰ます手順は複雑で深いはずなので、
				//       同じdepth ならnum_searched が大きい方を採用。
				//       https://link.springer.com/book/10.1007/978-3-319-09165-5の"Scalable Parallel DFPN Search"
				//       のwork の考えを流用。
				if (depth > max_depth) {
					max_depth = depth;
					max_depth_num_searched = child_entry_num_searched;
					max_pv = std::move(tmp_pv);
					for (int i = 0; i < max_pv.size(); ++i) {
						const auto& tmpMove = max_pv[i];
						//std::cout << "[And] max_pv[" << i << "] = " << tmpMove.toUSI() << std::endl;
					}
				}
				else if (depth == max_depth && child_entry_num_searched > max_depth_num_searched) {
					max_depth_num_searched = child_entry_num_searched;
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

	//flush_debug_str();

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

void ParallelDfPn::_set_result(const TTEntry& entry) {
	_result_pn = entry.pn;
	_result_dn = entry.dn;
	_result_is_mate = (entry.pn == 0);
	_result_is_repeat = (entry.num_searched == REPEAT);
	_result_is_done = true;
}

void ParallelDfPn::_set_result(int pn, int dn, bool is_mate, bool is_repeat) {
	_result_pn = pn;
	_result_dn = dn;
	_result_is_mate = is_mate;
	_result_is_repeat = is_repeat;
	_result_is_done = true;
}

// 詰将棋探索のエントリポイント
template<bool shared_stop>
void ParallelDfPn::dfpn(Position & r, int64_t & searched_node, const int threadid) {
	// キャッシュの世代を進める

	//std::cout << "debug: DLSHOGI dfpn::dfpn" << std::endl;
	//std::cout << "info: " << get_option_str() << std::endl;
	//sync_cout
	//	<< "[id=" << threadid << ",ply=" << r.gamePly() << ",hand=" << r.hand(r.turn()).value() << ",bk=" << r.getBoardKey()
	//	<< ",sfen=" << r.toSFEN() << "]" << sync_endl;

	searched_node = 0;    // NOTE: 探索ノード数をreset
	if (!r.inCheck()) {
		// 1手詰みチェック
		Move mate1ply = r.mateMoveIn1Ply<mmin1ply_Additional>();
		if (mate1ply) {
			auto& child_entry = transposition_table->LookUpChildEntry<true, UpdateOnly>(r, mate1ply);
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
			_set_result(0, kInfinitePnDn, true, false);
			return;
		}
	}
	dfpn_inner<true, shared_stop, true>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, std::min(r.gamePly() + kMaxDepth, draw_ply), searched_node, threadid);
	//sync_cout << "info: dfpn done, threadid = " << threadid << ", done, " << get_now_str() << sync_endl;

	auto&& entry = transposition_table->LookUp<true, ReadOnly>(r);

	//cout << searched_node << endl;

	/*std::vector<Move> moves;
	std::unordered_set<Key> visited;
	dfs(true, r, moves, visited);
	for (Move& move : moves)
	cout << move.toUSI() << " ";
	cout << endl;*/


	entry.lock();
	//sync_cout
	//	<< "[done" << ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
	//	<< ",id=" << threadid << ",ply=" << r.gamePly() << ",sfen=" << r.toSFEN() << "]" << sync_endl;
	//// debug
	//if (threadid == 23) {
	//	std::stringstream ss;
	//	ss << "[flush start:" << threadid << "]";
	//	enqueue_inf_debug_str(ss.str());
	//	flush_inf_debug_str();
	//}

	_set_result(entry);

#ifdef DEBUG_PRINT_WHEN_SEARCH_DONE_20231101_0
	if (!get_result_is_mate()) {
		sync_cout
			<< "searchDone,id = " << threadid << ",ply=" << r.gamePly()
			<< ",pn=" << entry.pn << ",dn=" << entry.dn << ",depth=" << entry.depth
			<< ",hand=" << entry.hand.value() << ",num_searched=" << entry.num_searched << sync_endl;
	}
#endif
	//sync_cout << "info: threadid = " << threadid << ", retVal = " << bts(retVal) << sync_endl;
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


	//return retVal;
}

// 詰将棋探索のエントリポイント
template<bool shared_stop>
void ParallelDfPn::dfpn_andnode(Position & r, int64_t & searched_node, const int threadid) {
	// 自玉に王手がかかっていること

	// キャッシュの世代を進める
	//transposition_table->NewSearch();

	searched_node = 0;
	dfpn_inner<false, shared_stop, true>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, std::min(r.gamePly() + kMaxDepth, draw_ply), searched_node, -1 /* dummy threadid */);
	auto& entry = transposition_table->LookUp<false, ReadOnly>(r);

	entry.lock();
	//const auto&& retVal = (entry.pn == 0);
	_set_result(entry);
	entry.unlock();

	//return retVal;
}

template<bool or_node>
void ParallelDfPn::print_entry_info(Position & n) {
	TTEntry& entry = transposition_table->LookUp<or_node, false>(n);
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

template void ParallelDfPn::dfpn<true>(Position & r, int64_t & searched_node, const int threadid);
template void ParallelDfPn::dfpn<false>(Position & r, int64_t & searched_node, const int threadid);

template void ParallelDfPn::dfpn_andnode<true>(Position & r, int64_t & searched_node, const int threadid);
template void ParallelDfPn::dfpn_andnode<false>(Position & r, int64_t & searched_node, const int threadid);

#endif    // DFPN_PARALLEL5