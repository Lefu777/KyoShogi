#pragma once
#include <atomic>
#include <mutex>
#include <thread>
#include <sstream>

#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"

#include "config.hpp"
//#include "util.hpp"
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


#ifdef DFPN_PARALLEL3

// 置換表
namespace ns_dfpn {
	struct alignas(32) TTEntry {
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

		void lock();
		void unlock();

	private:
		std::atomic<bool> _mtx;
	};

	struct TranspositionTable {
		struct Cluster {
			TTEntry entries[512];    // NOTE: 割とcluster size 大きいね。これ最適化出来そう。
		};

		~TranspositionTable();

		TTEntry& LookUp(const Key key, const Hand hand, const uint16_t depth);

		TTEntry& LookUpDirect(Cluster& entries, const uint32_t hash_high, const Hand hand, const uint16_t depth);

		template <bool or_node>
		TTEntry& LookUp(const Position& n);

		// moveを指した後の子ノードのキーを返す
		template <bool or_node>
		void GetChildFirstEntry(const Position& n, const Move move, Cluster*& entries, uint32_t& hash_high, Hand& hand);

		// moveを指した後の子ノードの置換表エントリを返す
		template <bool or_node>
		TTEntry& LookUpChildEntry(const Position& n, const Move move);

		// TODO
		//     : これしなくても、generation が異なれば、使われないはずだった気がするんだけどな。。。。
		//       まぁ一旦これでreset する。
		// 同じサイズの領域を再度確保することでクリア
		void Reset();

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
	void init();
	void new_search() { transposition_table.NewSearch(); }
	bool dfpn(Position& r, int64_t& searched_node, const int threadid);
	bool dfpn_andnode(Position& r, int64_t& searched_node, const int threadid);
	void dfpn_stop(const bool stop);
	Move dfpn_move(Position& pos);
	template <bool safe = false> std::tuple<std::string, int, Move> get_pv(Position& pos);

	template<bool or_node> void print_entry_info(Position& n);

	static void set_hashsize(const uint64_t size) {
		HASH_SIZE_MB = size;
	}
	static void set_draw_ply(const int draw_ply) {
		// WCSCのルールでは、最大手数で詰ました場合は勝ちになるため+1する
		ParallelDfPn::draw_ply = draw_ply + 1;
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
			<< " HASH_SIZE_MB=" << HASH_SIZE_MB;
		return ss.str();
	}

	void reset() {
		transposition_table.Reset();
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

	//int64_t searchedNode = 0;    // TODO: クリティカルセクションになり得る
private:
	template <bool or_node>
	void dfpn_inner(
		Position& n, const int thpn, const int thdn/*, bool inc_flag*/, const uint16_t maxDepth, int64_t& searchedNode, const int threadid
	);
	template<bool or_node, bool safe>
	int get_pv_inner(Position& pos, std::vector<Move>& pv);

	ns_dfpn::TranspositionTable transposition_table;
	std::atomic<bool> stop = false;
	int64_t maxSearchNode = 2097152;    // NOTE: 探索中はread only

	int kMaxDepth = 31;             // NOTE: 探索中はread only
	static int64_t HASH_SIZE_MB;    // NOTE: 探索中はread only
	static int draw_ply;            // NOTE: 探索中はread only
};


#endif    // DFPN_PARALLEL3