#pragma once
#include <atomic>
#include <mutex>
#include <thread>
#include <sstream>
#include <type_traits>

#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"

#include "config.hpp"
//#include "util.hpp"    // include するとなんかダメ。
#include "stop_watch.hpp"

// https://tadaoyamaoka.hatenablog.com/entry/2017/09/26/082125
// https://komorinfo.com/blog/df-pn-basics/
//     : OrNode
//        : pn(N) = min pn(N_m)
//        : dn(N) = sigma dn(N_m)
//     : AndNode
//        : pn(N) = sigma pn(N_m)
//        : dn(N) = min dn(N_m)
// https://komorinfo.com/blog/proof-piece-and-disproof-piece/
//     : 誤植があり、以下が正しいはず。Before は、与えられた持ち駒に差分を反映させる関数と言える。
//         : m が駒を取る手のとき：Before_n,m(h) := h - [mで取った駒]
//         : m が駒を打つ手のとき：Before_n, m(h) := h + [mで打った駒]
//     : 証明駒
//         : 最小を目指すべき
//         : 詰むことの十分条件とも言える。
//         ; OrNode での証明駒は、正解の手によって遷移した先の子ノードにおける証明駒 との差分を考えてやることで計算出来る。
//           正確に述べるなら、P_n = Before_n,m(P_m(n)) なる漸化式が成立する。
//         : AndNode での証明駒は、基本的には子ノードにおける証明駒の和集合であるが、遠隔王手が入るとそれだけではダメ。
//     : 反証駒
//         : 最大を目指すべき
//         : yet
// https://qhapaq.hatenablog.com/entry/2020/07/19/233054

// <point>
//     : OrNode, AndNode とは、
//       詰み  に関してはそれぞれOr, And で、
//       不詰みに関してはそれぞれAnd, Or である。
//         : pn, dn の計算が、OrNode でpn(N) = min pn(N_m)、AndNode でdn(N) = min dn(N_m) であるのもここから来ている。
//     : pn == 0 || dn == 0 が分かったら、直ちに 証明駒 || 反証駒 を計算して、置換表にset。


#ifdef DFPN_PARALLEL5

// 置換表
namespace ns_dfpn {
	constexpr uint32_t DISCARDED_GENERATION = -1;

	enum EntryState { PnIsZero, DnIsZero, EntryStateNum };

	template<EntryState es>
	inline bool should_update_hand(const Hand& new_hand, const Hand& old_hand) {
		if constexpr (es == PnIsZero) {
			// NOTE: 証明駒(詰ませるのに必要な駒) は少なければ少ないほど良い。
			if (old_hand.isEqualOrSuperior(new_hand)) {
				return true;
			}
			else {
				return false;
			}
		}
		else if constexpr (es == DnIsZero) {
			// NOTE: 反証駒(敵が持っていても詰まない駒) は多ければ多いほど良い。
			if (new_hand.isEqualOrSuperior(old_hand)) {
				return true;
			}
			else {
				return false;
			}
		}
		else {
			std::runtime_error("should_update() got unexpected EntryState.");
		}
	}

	struct alignas(8) TTEntry {
		// ハッシュの上位32ビット
		uint32_t hash_high;    // NOTE
		//     : 確かboard_key は64bit で、LookUp() の実装などと総合的に考えると、
		//       4294967296 個のhash (4294967296 * 256 個のentry) を使わない限り、board_key の上位32bit は
		//       hash へのアクセス時(hash key)には用いられない。
		//       なので、hash_key で同じcluster にアクセスした時(同じhash_keyの時)、更にこの上位32bit が一致するかを
		//       確認することで、hash 衝突はほぼほぼ起きなくなる。
		Hand hand; // 手駒（常に先手の手駒）
		int pn;
		int dn;
		uint16_t depth;
		uint16_t generation;
		uint32_t num_searched;    // NOTE
		//     : 一番初めは0 で初期化される。
		//     : 引き分けとか、最大手数(深さ制限)に達した時は、REPEAT が代入される。
		//     : 普段は、この局面をroot とした探索を何度実行したかを表す

		inline void lock();
		inline void unlock();

		//inline bool is_ok() const {
		//	return (_compute_sum() == _checksum);
		//}

		//inline void set_checksum() {
		//	// 32bit * 5 + 16bit * 2 = 32bit * 6 < 35bit
		//	_checksum = _compute_sum();
		//}

		//inline void discard_if_not_ok() {
		//	if (!is_ok()) {
		//		generation = DISCARDED_GENERATION;
		//	}
		//}

	private:
		//inline uint64_t _compute_sum() const {
		//	return (
		//		hash_high +
		//		hand.value() +
		//		pn +
		//		dn +
		//		depth +
		//		generation +
		//		num_searched
		//		);
		//}

		//uint64_t _checksum;    // atomic にしても良いかもね。

		std::atomic<bool> _mtx;
	};

	// NOTE
	//     : 現状、UpdateOnly に固有の機能は無く、ReadOnly の場合に、一部制限を掛けた形。
	//       ただ、ReadOnly と区別するという目的で意味があるし、boolean で制御するより見やすいはず。
	enum LookUpType { UpdateOnly, ReadOnly, LookUpTypeNum };

	struct TranspositionTable {
		struct Cluster {
			TTEntry entries[512];    // NOTE: 割とcluster size 大きいね。これ最適化出来そう。
		};

		~TranspositionTable();

		// HACK: UpdateOnly はちょっと違うな。Strict だったり、"update するのに向いてる(英語に)" とか
		//       そういう感じの名前の方が合ってるのでは？
		// @arg lut
		//     : UpdateOnly
		//           : 現状より汎用性を高める方向(superset)への更新ならして良い。
		//             その代わり、LookUp したentry の情報は、弱い可能性がある。
		//             ただし、その情報を用いても違法ではない。単に情報が弱い可能性があるだけ。
		//           : 逆に言えば、置換表にある、今の局面に関する情報をより強いものに更新したい場合はこっち。
		//     : ReadOnly
		//           : 情報の読み取り専用。情報としては一番汎用性の高いもの(上位集合?) を返すが、
		//             depth が異なる局面(即ち、異なる局面とみなす局面) の情報でも返してしまうので、
		//             今の局面の情報で問答無用で書き換えて良いものではない。
		template<LookUpType lut>
		TTEntry& LookUp(const Key key, const Hand hand, const uint16_t depth);

		template<LookUpType lut>
		TTEntry& LookUpDirect(Cluster& entries, const uint32_t hash_high, const Hand hand, const uint16_t depth);

		template <bool or_node, LookUpType lut>
		TTEntry& LookUp(const Position& n);

		// moveを指した後の子ノードのキーを返す
		template <bool or_node>
		void GetChildFirstEntry(const Position& n, const Move move, Cluster*& entries, uint32_t& hash_high, Hand& hand);

		// moveを指した後の子ノードの置換表エントリを返す
		template <bool or_node, LookUpType lut>
		TTEntry& LookUpChildEntry(const Position& n, const Move move);

		// TODO
		//     : これしなくても、generation が異なれば、使われないはずだった気がするんだけどな。。。。
		//       まぁ一旦これでreset する。
		// 同じサイズの領域を再度確保することでクリア
		void Reset();

		void Clear(int n_threads = 1);

		// 現在のsize と等しければ、何もしない
		void Resize(int64_t hash_size_mb);

		void NewSearch();

		void* tt_raw = nullptr;       // TODO: これ何？ -> see Resize()
		Cluster* tt = nullptr;        // TODO: これ何？ -> see Resize()
		int64_t num_clusters = 0;     // TODO: これ何？ -> see Resize()
		int64_t clusters_mask = 0;    // TODO
		//     : これ何？
		//       -> Resize() において、clusters_mask = num_clusters - 1; と代入される。
		//          つまり、hash size に応じてboard_key の下位bit の一部から、hash_key を生成する。
		uint16_t generation = 0;    // NOTE: NewSearch() が呼ばれなければスレッドセーフである
	};
}

// NOTE
//     : デフォルトコンストラクタ...?
class ParallelDfPn
{
public:
	// TODO
	//     :外部から与えられた置換表を勝手にNewSearch すべきでない。
	//      廃止すべき。
	void __new_search() { transposition_table->NewSearch(); }    // 使うべきでない

	template<bool shared_stop = false>
	void dfpn(Position& r, int64_t& searched_node, const int threadid);

	template<bool shared_stop = false>
	void dfpn_andnode(Position& r, int64_t& searched_node, const int threadid);
	void dfpn_stop(const bool stop);
	Move dfpn_move(Position& pos);
	template <bool safe = false> std::tuple<std::string, int, Move> get_pv(Position& pos);

	template<bool or_node> void print_entry_info(Position& n);

	void set_tt(ns_dfpn::TranspositionTable* tt) { transposition_table = tt; }

	// ポインタをset
	void set_shared_stop(std::atomic<bool>* shared_stop) { _shared_stop = shared_stop; }

	// 値をset
	void set_shared_stop_value(bool shared_stop) {
		if (_shared_stop == nullptr) {
			throw std::runtime_error("_shared_stop == nullptr");
		}
		_shared_stop->store(shared_stop);
	}

	//void set_hashsize(const uint64_t size) {
	//	hash_size_mb = size;
	//}
	void set_draw_ply(const int ply) {
		// WCSCのルールでは、最大手数で詰ました場合は勝ちになるため+1する
		draw_ply = ply + 1;
	}
	void set_maxdepth(const int depth) {
		kMaxDepth = depth;
	}
	void set_max_search_node(const int64_t max_search_node) {
		maxSearchNode = max_search_node;
	}

	// NOTE: info string で初めても、nodes[空白][hoge] という文字列があると、探索数の欄にhoge が表示されてしまうっぽい。
	std::string get_option_str() const {
		std::stringstream ss;
		ss
			<< "nodes=" << maxSearchNode
			<< " depth=" << kMaxDepth
			<< " draw_ply=" << draw_ply
			;
		return ss.str();
	}

	void reset() {
		transposition_table->Reset();
	}

	static void _print_entry_info(ns_dfpn::TTEntry& entry) {
		std::cout << "EntryInfo: ===== start =====" << std::endl;
		std::cout << "EntryInfo: hash_high = " << entry.hash_high << std::endl;
		std::cout << "EntryInfo: hand = " << entry.hand.value() << std::endl;
		std::cout << "EntryInfo: pn = " << entry.pn << std::endl;
		std::cout << "EntryInfo: dn = " << entry.dn << std::endl;
		std::cout << "EntryInfo: depth = " << entry.depth << std::endl;
		std::cout << "EntryInfo: generation = " << entry.generation << std::endl;
		std::cout << "EntryInfo: num_searched = " << entry.num_searched << std::endl;
		std::cout << "EntryInfo: ===== done =====" << std::endl;
	}

	// 必ず、"探索開始前(before calling dfpn())"にユーザー側で呼び出すこと。
	void init_result() {
		_result_pn = -1;
		_result_dn = -1;
		_result_is_mate = false;
		_result_is_repeat = false;
		_result_is_done = false;
	}
	// 必ず、"探索終了後(except during execution of dfpn())"にユーザー側で呼び出すこと。
	int get_result_pn() const { return _result_pn; }
	int get_result_dn() const { return _result_dn; }
	bool get_result_is_mate() const { return _result_is_mate; }
	bool get_result_is_repeat() const { return _result_is_repeat; }
	bool get_result_is_done() const { return _result_is_done; }

private:
	template <bool or_node, bool shared_stop, bool is_root>
	void dfpn_inner(
		Position& n, const int thpn, const int thdn/*, bool inc_flag*/, const uint16_t maxDepth, int64_t& searchedNode, const int threadid
	);
	template<bool or_node, bool safe>
	int get_pv_inner(Position& pos, std::vector<Move>& pv);

	template<bool shared_stop> inline bool _should_stop() const {
		if constexpr (shared_stop) {
			return _shared_stop->load();
		}
		else {
			return stop;
		}
	}

	//// result
	int _result_pn = -1;
	int _result_dn = -1;
	std::atomic<bool> _result_is_mate = false;
	std::atomic<bool> _result_is_repeat = false;
	std::atomic<bool> _result_is_done = false;    // result というよりstate だけど。。。
	// 必ず、"探索終了直後"に内部的に呼び出している。
	void _set_result(const ns_dfpn::TTEntry& entry);
	void _set_result(int pn, int dn, bool is_mate, bool is_repeat);

	//// member variable
	ns_dfpn::TranspositionTable* transposition_table = nullptr;
	std::atomic<bool> stop = false;
	std::atomic<bool>* _shared_stop = nullptr;    // 置換表を共有する時に使用
	int64_t maxSearchNode = 2097152;    // NOTE: 探索中はread only

	int kMaxDepth = 31;             // NOTE: 探索中はread only
	// TODO
	//     : 以下2つを非static にする。
	//     : TransitionTable で直接hash_size を設定できるようにする。
	//int64_t hash_size_mb = 2048;    // NOTE: 探索中はread only
	int draw_ply = INT_MAX;            // NOTE: 探索中はread only
};

#endif    // DFPN_PARALLEL5