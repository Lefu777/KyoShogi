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
//#include "util.hpp"    // include ����ƂȂ񂩃_���B
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
//     : ��A������A�ȉ����������͂��BBefore �́A�^����ꂽ������ɍ����𔽉f������֐��ƌ�����B
//         : m ���������̂Ƃ��FBefore_n,m(h) := h - [m�Ŏ������]
//         : m �����ł�̂Ƃ��FBefore_n, m(h) := h + [m�őł�����]
//     : �ؖ���
//         : �ŏ���ڎw���ׂ�
//         : �l�ނ��Ƃ̏\�������Ƃ�������B
//         ; OrNode �ł̏ؖ���́A�����̎�ɂ���đJ�ڂ�����̎q�m�[�h�ɂ�����ؖ��� �Ƃ̍������l���Ă�邱�ƂŌv�Z�o����B
//           ���m�ɏq�ׂ�Ȃ�AP_n = Before_n,m(P_m(n)) �Ȃ�Q��������������B
//         : AndNode �ł̏ؖ���́A��{�I�ɂ͎q�m�[�h�ɂ�����ؖ���̘a�W���ł��邪�A���u���肪����Ƃ��ꂾ���ł̓_���B
//     : ���؋�
//         : �ő��ڎw���ׂ�
//         : yet
// https://qhapaq.hatenablog.com/entry/2020/07/19/233054

// <point>
//     : OrNode, AndNode �Ƃ́A
//       �l��  �Ɋւ��Ă͂��ꂼ��Or, And �ŁA
//       �s�l�݂Ɋւ��Ă͂��ꂼ��And, Or �ł���B
//         : pn, dn �̌v�Z���AOrNode ��pn(N) = min pn(N_m)�AAndNode ��dn(N) = min dn(N_m) �ł���̂��������痈�Ă���B
//     : pn == 0 || dn == 0 ������������A������ �ؖ��� || ���؋� ���v�Z���āA�u���\��set�B


#ifdef DFPN_PARALLEL4

// �u���\
namespace ns_dfpn {
	constexpr uint32_t DISCARDED_GENERATION = -1;

	struct alignas(8) TTEntry {
		// �n�b�V���̏��32�r�b�g
		uint32_t hash_high;    // NOTE
		//     : �m��board_key ��64bit �ŁALookUp() �̎����ȂǂƑ����I�ɍl����ƁA
		//       4294967296 ��hash (4294967296 * 256 ��entry) ���g��Ȃ�����Aboard_key �̏��32bit ��
		//       hash �ւ̃A�N�Z�X��(hash key)�ɂ͗p�����Ȃ��B
		//       �Ȃ̂ŁAhash_key �œ���cluster �ɃA�N�Z�X������(����hash_key�̎�)�A�X�ɂ��̏��32bit ����v���邩��
		//       �m�F���邱�ƂŁAhash �Փ˂͂قڂقڋN���Ȃ��Ȃ�B
		Hand hand; // ���i��ɐ��̎��j
		int pn;
		int dn;
		uint16_t depth;
		uint16_t generation;
		uint32_t num_searched;    // NOTE
		//     : ��ԏ��߂�0 �ŏ����������B
		//     : ���������Ƃ��A�ő�萔(�[������)�ɒB�������́AREPEAT ����������B
		//     : ���i�́A���̋ǖʂ�root �Ƃ����T�������x���s��������\��

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

		//uint64_t _checksum;    // atomic �ɂ��Ă��ǂ������ˁB

		std::atomic<bool> _mtx;
	};

	struct TranspositionTable {
		struct Cluster {
			TTEntry entries[512];    // NOTE: ����cluster size �傫���ˁB����œK���o�������B
		};

		~TranspositionTable();

		TTEntry& LookUp(const Key key, const Hand hand, const uint16_t depth);

		TTEntry& LookUpDirect(Cluster& entries, const uint32_t hash_high, const Hand hand, const uint16_t depth);

		template <bool or_node>
		TTEntry& LookUp(const Position& n);

		// move���w������̎q�m�[�h�̃L�[��Ԃ�
		template <bool or_node>
		void GetChildFirstEntry(const Position& n, const Move move, Cluster*& entries, uint32_t& hash_high, Hand& hand);

		// move���w������̎q�m�[�h�̒u���\�G���g����Ԃ�
		template <bool or_node>
		TTEntry& LookUpChildEntry(const Position& n, const Move move);

		// TODO
		//     : ���ꂵ�Ȃ��Ă��Ageneration ���قȂ�΁A�g���Ȃ��͂��������C������񂾂��ǂȁB�B�B�B
		//       �܂���U�����reset ����B
		// �����T�C�Y�̗̈���ēx�m�ۂ��邱�ƂŃN���A
		void Reset();

		// ���݂�size �Ɠ�������΁A�������Ȃ�
		void Resize(int64_t hash_size_mb);

		void NewSearch();

		void* tt_raw = nullptr;       // TODO: ���ꉽ�H -> see Resize()
		Cluster* tt = nullptr;        // TODO: ���ꉽ�H -> see Resize()
		int64_t num_clusters = 0;     // TODO: ���ꉽ�H -> see Resize()
		int64_t clusters_mask = 0;    // TODO
		//     : ���ꉽ�H
		//       -> Resize() �ɂ����āAclusters_mask = num_clusters - 1; �Ƒ�������B
		//          �܂�Ahash size �ɉ�����board_key �̉���bit �̈ꕔ����Ahash_key �𐶐�����B
		uint16_t generation = 0;    // NOTE: NewSearch() ���Ă΂�Ȃ���΃X���b�h�Z�[�t�ł���
	};
}

// NOTE
//     : �f�t�H���g�R���X�g���N�^...?
class ParallelDfPn
{
public:
	void __new_search() { transposition_table->NewSearch(); }    // �g���ׂ��łȂ�
	template<bool shared_stop = false>
	bool dfpn(Position& r, int64_t& searched_node, const int threadid);
	template<bool shared_stop = false>
	bool dfpn_andnode(Position& r, int64_t& searched_node, const int threadid);
	void dfpn_stop(const bool stop);
	Move dfpn_move(Position& pos);
	template <bool safe = false> std::tuple<std::string, int, Move> get_pv(Position& pos);

	template<bool or_node> void print_entry_info(Position& n);

	void set_tt(ns_dfpn::TranspositionTable* tt) { transposition_table = tt; }

	// �|�C���^��set
	void set_shared_stop(std::atomic<bool>* shared_stop) { _shared_stop = shared_stop; }

	// �l��set
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
		// WCSC�̃��[���ł́A�ő�萔�ŋl�܂����ꍇ�͏����ɂȂ邽��+1����
		draw_ply = ply + 1;
	}
	void set_maxdepth(const int depth) {
		kMaxDepth = depth;
	}
	void set_max_search_node(const int64_t max_search_node) {
		maxSearchNode = max_search_node;
	}

	// NOTE: info string �ŏ��߂Ă��Anodes[��][hoge] �Ƃ��������񂪂���ƁA�T�����̗���hoge ���\������Ă��܂����ۂ��B
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

	//int64_t searchedNode = 0;    // TODO: �N���e�B�J���Z�N�V�����ɂȂ蓾��
private:
	template <bool or_node, bool shared_stop>
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

	//// member variable
	ns_dfpn::TranspositionTable* transposition_table = nullptr;
	std::atomic<bool> stop = false;
	std::atomic<bool>* _shared_stop = nullptr;    // �u���\�����L���鎞�Ɏg�p
	int64_t maxSearchNode = 2097152;    // NOTE: �T������read only

	int kMaxDepth = 31;             // NOTE: �T������read only
	// TODO
	//     : �ȉ�2���static �ɂ���B
	//     : TransitionTable �Œ���hash_size ��ݒ�ł���悤�ɂ���B
	//int64_t hash_size_mb = 2048;    // NOTE: �T������read only
	int draw_ply = INT_MAX;            // NOTE: �T������read only
};

#endif    // DFPN_PARALLEL4