#include <unordered_set>

#include "my_dfpn_parallel4.hpp"

#include "debug_string_queue.hpp"


#ifdef DFPN_PARALLEL4

//constexpr int skipSize[]  = { 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
//constexpr int skipPhase[] = { 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

constexpr int skipSize[]  = { 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
constexpr int skipPhase[] = { 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

//// debug
// �ȉ��̃n�b�V���̋ǖʂ�pn ��0 ���͂��̉\��������ϐ� �ɏ㏑�����ꂽ�ꍇ��queue_debug_str() ����B
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

// skip �ړIonly
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

// debug ���ɏo�͂������
//#define THREADID_COND (threadid == 1)


//// Impl
using namespace std;
using namespace ns_dfpn;

// MateMoveIn1Ply<>() ��additional
constexpr bool mmin1ply_Additional = true;

// 1 + epx
// https://link.springer.com/chapter/10.1007/978-3-319-09165-5_12
constexpr float EPS = 0;
constexpr float _EPS_PLUS_ONE = 1 + EPS;

// 1 �� 3 �� 4 �����x��������
constexpr float THPN_MULTIPLY = 2;
constexpr float THDN_MULTIPLY = 2;

// TODO
//     : ����mutex ���l�����邱�Ƃ��A����Cluster �ɃA�N�Z�X���邱�Ƃ̕K�vor�K�v�\�� �����Ŗ�����΂Ȃ�Ȃ��B
//       -> �����A����mutex ���l�����邱�Ƃ��A����Cluster �ɃA�N�Z�X���邱�Ƃ̏\�������ɂȂ��Ă͂Ȃ�Ȃ��B
//       -> �����A�C�ӂ�Cluster �ɑ΂��āA����Cluster �ɃA�N�Z�X�o����mutex ����ӂɒ�܂�Ȃ���΂Ȃ�Ȃ��B
//       -> �����ApositinoMutex �p��bit mask ��boardKey�̉���P bit�A
//          Cluster �ɂ��Ă�hask key �p��bit mask ��boardKey�̉���C bit �Ƃ���ƁAC >= P �𖞂����Ȃ���΂Ȃ�Ȃ�(�͂�)�B
constexpr uint64_t POS_MUTEX_NUM = 65536; // must be 2^n
std::mutex g_pos_mutexes[POS_MUTEX_NUM];

// NOTE: thread safe (https://yohhoy.hatenablog.jp/entry/2013/12/15/204116)
// mutex
inline std::mutex& get_position_mutex(const Position* pos)
{
	return g_pos_mutexes[pos->getKey() & (POS_MUTEX_NUM - 1)];
}



// NOTE
//     : REPEAT ���āA����� or �ő�萔�Œ��f �̏ꍇ��set ����āA������𑼂Ƌ�ʂ���ׁH
//int64_t ParallelDfPn::HASH_SIZE_MB = 2048;
//int ParallelDfPn::draw_ply = INT_MAX;
const constexpr uint32_t REPEAT = UINT_MAX - 1;

// --- �l�ݏ����T��

void ParallelDfPn::dfpn_stop(const bool stop)
{
	this->stop = stop;
}

// NOTE: �܂��ňȉ��̃R�����g�̒ʂ�ŁA�l���������ł��蓾��w�����S�ė񋓂���B
// �l�����G���W���p��MovePicker
namespace ns_dfpn {
	// ����entry��lock����B
	void TTEntry::lock() {
#ifdef DEBUG_LOCK_SAFE
		stopwatch_t sw;
		bool over_time_lim = false;
		sw.start();
#endif
		// NOTE: fales �̎���true �ւƃA�g�~�b�N�ɏ����������o������A���̏u�Ԃɔr���I�ȃ��b�N���l���ł��Ă�B
		// �T�^�I��CAS lock
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

	// ����entry��unlock����B
	// lock�ς݂ł��邱�ƁB
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


	// NOTE: �C�ӂ̋ǖʂɂ����鉤����|����w����̐��́AMaxCheckMoves �ȉ��ł���H
	const constexpr size_t MaxCheckMoves = 91;

	template <bool or_node>
	class MovePicker {
	public:
		explicit MovePicker(const Position& pos) {
			if (or_node) {
				// NOTE: ���̊֐��ɂ���āAmoveList_ ��begin, last_ ��end �ƂȂ�B
				last_ = generateMoves<CheckAll>(moveList_, pos);
				if (pos.inCheck()) {
					// ���ʂ�����̏ꍇ�A������肩������������𐶐�
					ExtMove* curr = moveList_;
					while (curr != last_) {
						if (!pos.moveIsPseudoLegal<false>(curr->move)) {
							// NOTE
							//     : �񍇖@��𔭌������ꍇ�́A��������v�f�������ė��ď㏑������B
							//       ���̎�end -= 1 ����A�㏑�������z�͂��̎��Ō��؂����B
							curr->move = (--last_)->move;
						}
						else {
							++curr;
						}
					}
				}
			}
			else {
				last_ = generateMoves<Evasion>(moveList_, pos);    // NOTE: �R�C�c��pseudoLegal ��Ԃ�
				// �ʂ̈ړ��ɂ�鎩�E��ƁApin����Ă����̈ړ��ɂ�鎩�E����폜
				ExtMove* curr = moveList_;
				const Bitboard&& pinned = pos.pinnedBB();    // NOTE: ��Q�Ƃ���E�Ӓl�Q�ƂɕύX
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

// �u���\
TranspositionTable::~TranspositionTable() {
	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}
}

// TODO
//     : ����key ��position mutex �l�������lock �o����͂��B
TTEntry& TranspositionTable::LookUp(const Key key, const Hand hand, const uint16_t depth) {
	auto& entries = tt[key & clusters_mask];    // TODO: ����f�[�^���������ˁB
	uint32_t hash_high = key >> 32;

#ifdef DEBUG
	if (hash_high == HASH_HIGH_DEBGU_TARGET3) {
		std::cout << "[" << key << "]" << std::endl;
	}
#endif

	return LookUpDirect(entries, hash_high, hand, depth);
}

// TODO
//     : �ꉞ�Aentries �ɃA�N�Z�X����ׂ�key ��position_mutex �擾����΃X���b�h�Z�[�t�����A
//       LookUpDirect ���̂́Aentries �ɒu���\����A�N�Z�X����ׂ�key ��m��Ȃ��̂ł��̊֐��ł�lock�o���Ȃ��B
// NOTE
//     : ����hand �ɂ́ALookUp<bool or_node>() ��ʂ���OR �Ȃ��ԑ��̎�����AAND �ɂ͓G���̎�����n�����(�͂�)�B
//       -> �v����ɁAhand �͋l�܂����̎�����n�����B
TTEntry& TranspositionTable::LookUpDirect(
	Cluster& entries, const uint32_t hash_high, const Hand hand, const uint16_t depth
) {

	int max_pn = 1;
	int max_dn = 1;

	// ���������ɍ��v����G���g����Ԃ�
	for (size_t i = 0; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
		TTEntry& entry = entries.entries[i];

		entry.lock();

		// TODO
		//     : break ����return �Ŕ����钼�O�ł͕K��unlock()
		//       ����ȊO�͎g��Ȃ��Ȃ�����unlock();
		//     : �ǂ���unlock() ����ĂȂ�������A���S�����e�X�g����΂ǂ����̓f�b�h���b�N�N�����Ă����ł���B
		//       ���̎��ɁACAS lock �̏��ő҂����Ԓ����ꍇ�Ɍx���o���΂悢�B
		// NOTE
		//     : ���̐���Ɉ�v���Ȃ����̂́A�S�~�ƌ��Ȃ��H(���p�����ɏ㏑������H)
		if (generation != entry.generation) {
			// NOTE: �����͑S����薳���͂��B�B�B
			// ��̃G���g�������������ꍇ
			entry.hash_high = hash_high;
			entry.depth = depth;
			entry.hand = hand;
			// TODO
			//     : �ň��̏ꍇ��z�肵��max ��o�^���Ă����H
			//       ���Ƃ�����r���܂ł���Ȃ��ĂȂ�őS�̂̍ő�l�ɂ��Ȃ��́H���̕ӂ�͎G�ŗǂ����Ă��ƁH
			entry.pn = max_pn;
			entry.dn = max_dn;
			entry.generation = generation;
			entry.num_searched = 0;

			entry.unlock();
			return entry;

		}

		// TODO
		//     : hash �Ɛ��オ��v����̂͐�΂ɕK�v�H generation ���i�ރ^�C�~���O�́H(see dlshogi)
		if (hash_high == entry.hash_high && generation == entry.generation) {    // NOTE: �����ǖʂƌ��Ȃ��K�v����
			if (hand == entry.hand && depth == entry.depth) {
				entry.unlock();
				// key�����v����G���g�����������ꍇ
				// �c��̃G���g���ɗD�z�֌W�𖞂����ǖʂ�����ؖ��ς݂̏ꍇ�A�����Ԃ�
				int debug_i_tmp = i;
				for (i++; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
					TTEntry& entry_rest = entries.entries[i];
					entry_rest.lock();
					if (generation != entry_rest.generation) {
						// TODO
						//     : ����continue ����_���Ȃ�ł����H
						//       -> �O����ׂ��Ă����̂ŁA�قȂ�generatino �����ꂽ��A����͍���generation ����O�ŏI�������Ƃ������ƁB
						//          ����䂦����ȏ㑖�����Ă����ʁB
						entry_rest.unlock();
						break;
					}
					// TODO
					//     : ��������Ԃ�ꍇ�A����depth ���قȂ�Ȃ珑��������ׂ��łȂ��B�B�B�H
					if (hash_high == entry_rest.hash_high) {
						// NOTE
						//     : �l�܂������A�c���entry �̎�����ȏ�Ɏ�����������Ă���Ȃ��ʌ݊��ŁA�l�ނ͂��A�I�Șb�H
						//       -> key �ŃA�N�Z�X���Ă�̂ŁA�Ֆʂ����Ō��Ă���v����Ƃ͌�����ȁH���̂�����ǂ��Ȃ��Ă�H
						//     : (�l�ނ��A���A���������ʌ݊��Ȃ̂ł����Ԃ��A�Ƃ������B�܂�A����entry_rest �̋ǖ�(�̎�����)�̕����ėp���������B)
						if (entry_rest.pn == 0) {    // NOTE:  �l�݂��ؖ��ς�
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
								//     : ��������Ԃ�entry �́A�X�V�͂ނ�݂₽��ɂ��Ă͂����Ȃ��͂��B
								//       ��������́A���ǖʂ̎�������ėp���̍���������̋ǖ�(��Ԕėp��������)�̏����Q�Ƃ��Ă�B
								//       �X�V�͕K���A�����̔ėp��or���󒴉߂̔ėp�������l�ɏ㏑�������͂��Ȃ̂ŁA
								//       ���ǖʂ̔ėp���Ɠ����̔ėp���ŏ㏑�����Ă��܂��A��Ԕėp���������������̔ėp���������Ă��܂��\��������B
								return entry_rest;
							}
						}
						else if (entry_rest.dn == 0) {    // NOTE: �s�l�ݏؖ��ς�
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
				//     : ��苭�����Ƃ�������entry ��������΁A��Ɍ��������z��Ԃ�
				//       query �Ƃ��ė^����ꂽ�ǖ�(�̎�����)���ėp���̍���entry �͌�����Ȃ������B
				//       -> ���̏ꍇ�A����entry �Ȃ�A�X�V���鎞�ɁA�X�V���Ă͂Ȃ�Ȃ������͑��݂��Ȃ��͂��B�B�B
				//          (��X�V�͕K���A�����̔ėp��or���󒴉߂̔ėp�������l�ɏ㏑�������͂��B
				//             �Ȃ̂ŁA�����Ԕėp���̍���entry �̔ėp��������������ɍX�V���邱�Ƃ͂Ȃ��B)
				//          
				return entry;
			}
			// TODO
			//     : ����hash ������v���āA������̗D�z�֌W�𖞂����ؖ��ς݋ǖʂ�����΂����Ԃ��A�ɂ����ɁA
			//       generation ����v���邱�Ƃ��m�F���Ă�́H
			//       -> L107 ��if (generation != entry.generation) { ����̐߂����Ă�������悤�ɁA
			//          �O�̌Â�����̒T�����ʂ͗p�����A���̐���̒T�����ʂ݂̂�p���邩��B
			// TODO
			//     : depth ���قȂ��Ă��l�݂Ȃ�entry ��Ԃ����Ⴄ�ƁA
			//       �Ⴆ��depth=6 �Ȃ�l�݂��������ǁAdepth=10����depth�����ň��������ɂȂ��Ă��܂����A
			//       depth=6 �ŋl�݂������̂ɁA�ő�萔���������ŕs�l��(dn=inf) �ɏ����������Ă��܂��B
			//     : �����Aroot node �ł͂��ꂪ�N���Ȃ��͂��B���̂Ȃ�depth ���قȂ��Ă��Ԃ��̂͋l�݂��ؖ�����Ă���ꍇ�݂̂ł���A
			//       root node �ŋl�݂��ؖ������΂قڂقڒ����ɒT���͏I������͂�������ł���B
			//       -> �Ȃ̂ŁA�N���Ȃ��͂��͌����߂������ǁA�m���͂��Ⴂ�ƌ�����B
			// TODO
			//     : ��������Ԃ�ꍇ�A����depth ���قȂ�Ȃ珑��������ׂ��łȂ��B�B�B�H
			// �D�z�֌W�𖞂����ǖʂɏؖ��ς݂̋ǖʂ�����ꍇ�A�����Ԃ�
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
			// TODO: ���̈ȉ��̏ꍇ�ɍő�l���X�V����̂��H
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
	//     : �T��������ԏ��Ȃ��ǖʒ��Ahash �ɒ��߂Ă������l���Ⴂ�̂ŁA�T��������ԏ��Ȃ�entry �������Ēׂ��B
	//       ���̍ۂɁA�l�݂��ؖ��ł��Ă��Ȃ��ǖʂ���D��I�ɒׂ��悤�ɂ���B
	//cout << "hash entry full" << endl;
	// ���v����G���g����������Ȃ������̂�
	// �Â��G���g�����Ԃ�
	TTEntry* best_entry = nullptr;
	uint32_t best_num_searched = UINT32_MAX;         // NOTE: num_searched �̍ŏ��l�H
	TTEntry* best_entry_include_mate = nullptr;
	uint32_t best_num_searched_include_mate = UINT32_MAX;    // NOTE
	//     : �l�݋ǖʂ݂̂�num_searched �ŏ��l�H
	//       ���O�I�ɂ͈Ⴄ���ۂ����ǁB�B�B

// NOTE
//     : ����㏑������\���̂���entry ��lock �͉�����ׂ��łȂ��͂��B
//       ��U������āA���̊Ԃɉ��l�̂���ʂ̏�񂪏������܂ꂽ�Ƃ��ɂ����ׂ����ƂɂȂ�B
//       �܂��\���傫�Ȓu���\������Ζ��Ȃ��͂��B
//       �����A30�X���b�h���炢�ŒT������̂ɁAPUCT��64GB���炢�͐H���͂��Ŏc���64GB�ŏ\���傫�����Ƃ����̂͋^�₪�c��B
	for (auto& entry : entries.entries) {
		entry.lock();
		if (entry.pn != 0) {
			if (best_num_searched > entry.num_searched) {
				if (best_entry != nullptr) {
					best_entry->unlock();    // ����������\���͖����Ȃ����̂ŁA��O��best�͉��
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
					best_entry_include_mate->unlock();    // ����������\���͖����Ȃ����̂ŁA��O��best�͉��
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
	//     : �l�݂��ؖ��ł��Ă���ǖʂ̉��l�͍����̂ŁA
	//       �l�݂��ؖ��ł��Ă��Ȃ��ǖʂ�����������ɂ̂݁A�l�݂��ؖ��ł��Ă���ǖʂ�ׂ��B
	if (best_entry == nullptr) {
		best_entry = best_entry_include_mate;
	}
	else {
		// best_entry_include_mate �́A����best_entry �Ƃ��Ďg���Ȃ���Έꐶ�������Ȃ��̂ł����ŉ��
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
	// NOTE: ����LookUp<>() ��LookUp() �̃��b�p�[
	auto& retVal = LookUp(n.getBoardKey(), or_node ? n.hand(n.turn()) : n.hand(oppositeColor(n.turn())), n.gamePly());
	return retVal;
}

// TODO
//     : lock �֘A�ǂ����邩�B
//       -> entries �ɏ������ނ��Ƃ͖����͂��ŁA�����܂ŃA�h���X��ǂݎ�邾���ł���A�T�����ɃA�h���X�������������鎖�͖����͂��ŁA
//          �N���e�B�J���Z�N�V�����ɂ͂Ȃ蓾�Ȃ��͂��B 
// move���w������̎q�m�[�h�̃L�[��Ԃ�
template <bool or_node>
void TranspositionTable::GetChildFirstEntry(const Position& n, const Move move, Cluster*& entries, uint32_t& hash_high, Hand& hand) {
	// ���͏�ɐ��̎��ŕ\��
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
	//     : ����f�[�^��������H
	//     : �܂��ACluster tt[]; �ł���B
	//       tt ��Cluster ��ǂނ̂ƁAentry.pn = 0 �Ƃ�����̂́A�r�����䂳��Ă��Ȃ��B(entry.lock() ��tt ����Cluster ��ǂނ̂ɑ΂��Č��ʂ𔭊����Ȃ�)
	//       
	entries = &tt[key & clusters_mask];
	hash_high = key >> 32;
	//return key;
}

// TODO
//     : lock �֘A�ǂ����邩�B
//       LookUpDirect �Ă񂶂���Ă�̂ŁAauto&& retVal = LookUpDirec() �Ƃ��đO����͂��ׂ��H
//       �Ă����ꂩ�A���ꂷ��ɂ�n.getBoardKeyAfter(move); �Ȃ邱�̃m�[�h��key ���K�v�B
//       GetChildFirstEntry() ����Ԃ��Ă��炤���H
// move���w������̎q�m�[�h�̒u���\�G���g����Ԃ�
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

// TODO: ��˂��牤�݂����ɁA�ŏ����畨����������Ɋm�ۂ�����
// �̈���Ċm�ۂ��邱�Ƃ�reset ���Ă�B�񐄏��B
void TranspositionTable::Reset() {
	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}

	// NOTE
	//     : ���Ɍ^�͎w�肹���ɁA��������(new_num_clusters * sizeof(Cluster) + CacheLineSize) * (1) byte �̗̈���m��
	// TODO
	//     : CacheLineSize �Ƃ́H
	//     : ((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1)) ���ĉ����Ă�H
	tt_raw = std::calloc(num_clusters * sizeof(Cluster) + CacheLineSize, 1);
	tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));
	clusters_mask = num_clusters - 1;
}

void TranspositionTable::Resize(int64_t hash_size_mb) {
	if (hash_size_mb == 16) {
		// TODO: �ǂ��������ƁH
		hash_size_mb = 4096;
	}
	// NOTE
	//     : msb = Most Significant Bit = �ŏ��bit
	//       �܂�A�m�ۂł�����E�e�ʂɌ���Ȃ��߂Â��Ahash �̏c�̗�̃T�C�Y��2��n�� �ɂ������A���Ċ������ȁB
	//       ���炭hash �̃A�N�Z�X����key ���v�Z����ۂ�mask(=clusters_mask) ��popcnt == 1 �ł���P����bit �ɏo���邩�炩�ȁH
	int64_t new_num_clusters = 1LL << msb((hash_size_mb * 1024 * 1024) / sizeof(Cluster));
	if (new_num_clusters == num_clusters) {
		return;
	}

	num_clusters = new_num_clusters;
	// TOOD
	//     : ����A���ʂ�if �Ŕ��肷�ׂ��ȋC�͂���B
	//       ����Ahash_size_mb = 512mb �������ƃA�E�g�B���炭�ˁB
	assert(num_clusters >= POS_MUTEX_NUM && "expected num_clusters >= POS_MUTEX_NUM, but got num_clusters < POS_MUTEX_NUM");

	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}

	// NOTE: OverFlow �Ƃ��������Ƃ���h�����߂ɁA�����Ċ|���Ċ����Ă�
	std::cout << "info string alloc_size=" << (sizeof(Cluster) * (new_num_clusters / 1024)) / 1024 << "mb" << std::endl;

	// NOTE
	//     : ���Ɍ^�͎w�肹���ɁA��������(new_num_clusters * sizeof(Cluster) + CacheLineSize) * (1) byte �̗̈���m��
	// TODO
	//     : CacheLineSize �Ƃ́H
	//     : ((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1)) ���ĉ����Ă�H
	tt_raw = std::calloc(new_num_clusters * sizeof(Cluster) + CacheLineSize, 1);
	if (tt_raw == nullptr) {
		// �Ȃ�bad_alloc �o�Ȃ������̂ł���ŁB
		std::cout << "Error: bad_alloc! Resize() failed to calloc" << std::endl;
		exit(1);
	}

	tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));
	clusters_mask = num_clusters - 1;
}

void TranspositionTable::NewSearch() {
	++generation;
	// TODO: �Ȃ���������́H���߂�generation = -1 ���Ă��Ƃ����H
	if (generation == 0) generation = 1;
}

static const constexpr int kInfinitePnDn = 100000000;

// ����̎w���肪�ߐډ��肩
FORCE_INLINE bool moveGivesNeighborCheck(const Position& pos, const Move& move)
{
	const Color them = oppositeColor(pos.turn());
	const Square ksq = pos.kingSquare(them);

	const Square to = move.to();

	// �G�ʂ�8�ߖT
	if (pos.attacksFrom<King>(ksq).isSet(to))
		return true;

	// �j�n�ɂ�鉤��
	if (move.pieceTypeTo() == Knight)
		return true;

	return false;
}

// TODO
//     : ���̂��̎����ŗǂ��̂��������B
//       �ʂɁA����ȏ�Ɏ�����𑝂₵�Ă��l�܂Ȃ����ĕۏ؂͖����Ȃ����H
// ���؋���v�Z(�����Ă��鎝������ő吔�ɂ���(���̎������������))
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
//     : lock�֘A�ǂ�����˂�B
//       LookUp() �Ŗ����entry �Ƀo�`�R����������܂����ȁBposition n ��key ��position mutex �ł����܂��H
//       
// @arg or_node
//     : OrNode �ł���B
// @arg shared_stop
//     : ��~��_shared_stop ���g��
// @arg thpn, thdn
//     : ��ԏ��߂�root �ł�inf ���n�������ۂ��B
//     : ���̋ǖʂɓ����Ă��钼�O�ɂ����āA
//       Position n �̌Z��ǖ� �̓��A2�Ԗڂɗǂ��ǖʂ�pn, dn
//       ��2�Ԗڂɗǂ��ǖʂƂ́A
//         OrNode �Ȃ�2�Ԗڂ�pn ���������ǖʁA
//         AndNode �Ȃ�2�Ԗڂ�dn ���������ǖ� �̂��Ƃ��w���B
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
		// NOTE: �[�������ɒB������A�s�l�݂Ɣ���
		if (n.gamePly() + 1 > maxDepth) {
			entry.lock();
			// NOTE
			//     : ����́Adepth ���قȂ��Ă��A���������֌W�𖞂�����entry ��Ԃ��Ă��܂��̂ŁA
			//       �l�݋ǖʂ��ő�萔���������ɂ��s�l�݂ŏ㏑�����Ă��܂��̂�h���B
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				// TODO20231026: �㏑���̂�
				// NOTE: ���̏ꍇ�A���؋�������Ȃ��B�����čő�萔���������Ȃ񂾂��́B
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				entry.num_searched = REPEAT;    // TODO: REPEAT ���ĉ��H

			}
			entry.unlock();
			return;    // NOTE: serach_result = ���f(depth limit)
		}
	}

	// NOTE
	//     : hash �����S�Ɉ�v���Ă���̂ɔՖʂ��Ⴄ���Ƃ͖����Ɖ��肵���ꍇ�A
	//       pn=inf �ɏ�����������o�O���N����̂́A�ǖʂ͓����Ȃ̂Ɏ萔�ɂ���ċl�ݕs�l�݂��ς��ꍇ�B
	//       �ȉ��̏ꍇ�́A�萔�ɂ�炸emtpy() �ł���̂Ŗ��Ȃ��B
	// if (n is a terminal node) { handle n and return; }
	MovePicker<or_node> move_picker(n);
	if (move_picker.empty()) {    // NOTE: �肪����ȏ㑱�����l�܂���ꖳ�� or �l�܂��ꂽ�̂œ������邵���Ȃ�
		// n����[�m�[�h
		entry.lock();

		if (or_node) {

			// �����̎�Ԃł����ɓ��B�����ꍇ�͉���̎肪���������A
#ifdef DFPN_MOD_V2
			if (entry.pn != 0) {
#else
			if (true) {
#endif
				// TODO20231026: �㏑���̂�
				entry.pn = kInfinitePnDn;
				entry.dn = 0;
				// ���؋�
				// �����Ă��鎝������ő吔�ɂ���(���̎������������)
				entry.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
			}
		}
		else {
#ifdef DFPN_MOD_V3
			if (entry.dn != 0) {
#else
			if (true) {
#endif
				// ����̎�Ԃł����ɓ��B�����ꍇ�͉������̎肪���������A
				// 1��l�߂��s���Ă��邽�߁A�����ɓ��B���邱�Ƃ͂Ȃ�
				// debug
				if (entry.hash_high == DEBUG_HASH_HIGH_TAG && entry.depth == 23) {
					std::stringstream ss;
					ss << "[pn:" << entry.pn << "," << entry.dn
						<< "," << entry.depth << "," << entry.hand.value() << "," << entry.generation << "]";
					enqueue_debug_str(ss.str());
				}
				// TODO20231026: �㏑���̂�
				entry.pn = 0;
				entry.dn = kInfinitePnDn;
			}
		}
		entry.unlock();
		return;    // NOTE: serach_result = �s�l�� || �l��
	}

	//// NOTE
	////     : ��U�A2 or 3��l�� �T��������B
	// �V�K�ߓ_�ŌŒ�[���̒T���𕹗p
	entry.lock();
	if (entry.num_searched == 0) {
		entry.unlock();

		if (or_node) {
			// 3��l�݃`�F�b�N
			Color us = n.turn();
			Color them = oppositeColor(us);

			StateInfo si;
			StateInfo si2;

			const CheckInfo ci(n);
			for (const auto& ml : move_picker) {    // NOTE: OrNode �ł̎w�����T��
				const Move& m = ml.move;

				n.doMove(m, si, ci, true);

				// �����̃`�F�b�N
				if (n.isDraw(16) == RepetitionWin) {
					// NOTE: �A������̐����
					// �󂯑��̔�������
					n.undoMove(m);
					continue;
				}

				// TODO
				//     : �r����entry2 �̒l������������ꂽ�ꍇ�ɁA�����s�s�����邩�ˁH
				//       -> ���[�A���ꂾ�A�������������鎞�ɑS���̏�������������󂶂�Ȃ�����A
				//          �r���Œu�����ꂽ�ꍇ�A��{�I�ɂ�hash_high & depth vs ���̑��̏�� �ňقȂ�ǖʂ̏�񂪈��entry �ɏ������܂�Ă��܂��B
				//          ����Ō����ƁAroot node ��entry �������Ȃ�B������lock ���Ȃ��Ƃ����Ȃ��Ȃ�B
				//          ����A���or���i�߂����ɁAroot �ƑS������key �ɂȂ�\�����Ă��邩�ȁH
				//          ���݂ɁA4��i�߂�������œ���key �ǂ��납�����ǖʂɂȂ蓾��̂Ńf�b�h���b�N�ɂȂ�̂Ń_���B
				auto& entry2 = transposition_table->LookUp<false>(n);

				// ���̋ǖʂł��ׂĂ�evasion������
				MovePicker<false> move_picker2(n);

				if (move_picker2.size() == 0) {
					// NOTE: ���ɓ�����肪�����̂ŁA����̎w����m �͋l�܂����ł���B
					// 1��ŋl��
					n.undoMove(m);

					// TODO
					//     : kInfinitePnDn + 1 ��+1 ���闝�R�͉��H
					//       -> get_pv_inner �Ƃ��ł��g���ĂāA�g���Ă�Ƃ��댩�������A
					//          AndNode �ł�+1, OrNode �ł�+2 ���ۂ��B�B�B?
					//       -> �����ƌ����ƁA+1 �̎��A���̋ǖʂ͋l�݋ǖʂ��̂��́H
					//          �ق�ł�����+2 �̎��A1��l�ߋǖʂ��ȁH

					// NOTE
					//     : hash_high �͈�x�������܂ꂽ��A�V��������or�u�� ����Ȃ�����͏������܂�Ȃ��B
					//       �Ȃ̂ŁA���A�����Adebug ��p�������Ulock �͖����ŁB
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
						// TODO20231026: �㏑���̂�
						entry2.pn = 0;
						entry2.dn = kInfinitePnDn + 1;
					}
					entry2.unlock();
					entry.lock();
#ifdef DFPN_MOD_V3
					// TODO20231026
					//     : dn = 0�̏ꍇ�͏㏑�����Ă��ǂ��̂ł́H
					//       �����āA�ő�萔�ŕs�l�ݔ��肳�ꂽ�ǖʂ��A���Z�萔�Ŕ����ł�����l�݂ɕς�����ŁB�B
					//       -> �t�Ƀ_���ȏꍇ���ĂȂ񂩂���H
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
						// TODO20231026: �㏑���̂�
						entry.pn = 0;
						entry.dn = kInfinitePnDn;

						// NOTE
						//     : ��`�ʂ�ɂ���Ȃ�A�l�񂾋ǖʂł���entry2 �ł̏ؖ��� = 0 �ŁA
						//       m �����ł�Ȃ�� "entry �ł̏ؖ��� = entry.hand + �ł�����" �ƍ����X�V
						// �ؖ����������
						entry.hand.set(0);

						// �ł�Ȃ�Ώؖ���ɉ�����
						if (m.isDrop()) {
							entry.hand.plusOne(m.handPieceDropped());
						}
						// ��肪�ꖇ�������Ă��Ȃ���ނ̐��̎�������ؖ���ɐݒ肷��
						if (!moveGivesNeighborCheck(n, m))
							entry.hand.setPP(n.hand(n.turn()), n.hand(oppositeColor(n.turn())));
					}
					entry.unlock();
					return;    // NOTE: serach_result = �l��
				}

				// TODO
				//     : ����+2 �H
				//     : move_picker ����ɃR�b�`�`�F�b�N���ׂ��ł́H�H
				// NOTE: �萔���E��˔j�����̂ŁA�s�l�݈���
				if (n.gamePly() + 2 > maxDepth) {
					n.undoMove(m);

					entry2.lock();
#ifdef DFPN_MOD_V2
					if (entry2.pn != 0) {
#else
					if (true) {
#endif
						// TODO20231026: �㏑���̂�
						entry2.pn = kInfinitePnDn;
						entry2.dn = 0;
						entry2.num_searched = REPEAT;
					}
					entry2.unlock();

					continue;
				}

				// NOTE
				//     : ���͑J�ڂ���AndNode �ɋ���̂ŁA��ł��l�݂𓦂��w���肪����΁ANEXT_CHECK �ւƃW�����v����B
				const CheckInfo ci2(n);
				for (const auto& move : move_picker2) {    // NOTE: AndNode �ł̎w�����T��
					const Move& m2 = move.move;

					// ���̎w����ŋt����ɂȂ�Ȃ�A�s�l�߂Ƃ��Ĉ���
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
							// TODO: �ؖ����set ���ׂ��B
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
							//     : �㏑���̂�
							//       �����ł́A���̎Q�Ƃ͂����A���̋ǖʂ��l��(����1��l�߂ł���)���Ƃ�entry �ɕۑ��������B
							//       -> �܂�Adepth ��hand ��v�A�Ⴕ���͐V����entry �ɏ������ގ��݂̂��������B
							entry1.pn = 0;
							entry1.dn = kInfinitePnDn + 2;

						}
						entry1.unlock();
					}
					else {
						// �l��łȂ��̂ŁAm2�ŋl�݂𓦂�Ă���B
						n.undoMove(m2);
						goto NEXT_CHECK;
					}

					n.undoMove(m2);
				}

				// ���ׂċl��
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
					// TODO20231026: �㏑���̂�
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
					// TODO20231026: �㏑���̂�
					entry.pn = 0;
					entry.dn = kInfinitePnDn;
				}
				entry.unlock();

				return;

				// NOTE
				//     : ����T������OrNode �̎w����́A���ɂȂ�炩�̓�������������
			NEXT_CHECK:;
				n.undoMove(m);
				entry2.lock();
				if (entry2.num_searched == 0) {
					// TODO
					//     : ����pn, dn �����L�̒l�ɂȂ�H
					//       pn �͂܂�������C�����邪�Adn �͉���?
					//       ����A���ꂩ�AAndNode ������Apn = move_picker2.size() �Ȃ̂͗ǂ����B(��������A�q�m�[�h�W�J���Ȃ��Ƃ����Ȃ��̂ł́H)
					//       �����ǁAdn�� dn = 1; �Ƃ��ׂ�����Ȃ��́H���ʂɂ���Ȃ炳�H
					//     : ���������A����num_searched == 0 �̎������H
			    	// TODO20231026: �㏑���̂�
					entry2.num_searched = 1;
					entry2.pn = static_cast<int>(move_picker2.size());
					entry2.dn = static_cast<int>(move_picker2.size());
				}
				entry2.unlock();
			}
		}
		else {
			// 2��ǂ݃`�F�b�N
			StateInfo si2;
			// ���̋ǖʂł��ׂĂ�evasion������
			const CheckInfo ci2(n);
			for (const auto& move : move_picker) {    // NOTE: AndNode �̎w����ŁA�S�����ׂ�B
				const Move& m2 = move.move;

				// ���̎w����ŋt����ɂȂ�Ȃ�A�s�l�߂Ƃ��Ĉ���
				if (n.moveGivesCheck(m2, ci2))
					goto NO_MATE;

				n.doMove(m2, si2, ci2, false);

				// TODO
				//     : dlshogi �ɂȂ���āA
				//       template <bool Additional = true> Move mateMoveIn1Ply(); ��
				//       template <bool Additional = false> Move mateMoveIn1Ply(); �ɁB
				//       (���m�ɂ́Adlshogi ��Additional �������B)
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

						// TODO20231026: �㏑���̂�
						entry1.pn = 0;
						entry1.dn = kInfinitePnDn + 2;

						// �ؖ����������
						entry1.hand.set(0);

						// �ł�Ȃ�Ώؖ���ɉ�����
						if (move.isDrop()) {
							entry1.hand.plusOne(move.handPieceDropped());
						}
						// ��肪�ꖇ�������Ă��Ȃ���ނ̐��̎�������ؖ���ɐݒ肷��
						if (!moveGivesNeighborCheck(n, move))
							entry1.hand.setPP(n.hand(n.turn()), n.hand(oppositeColor(n.turn())));
					}
					entry1.unlock();
				}
				else {
					// �l��łȂ��̂ŁAm2�ŋl�݂𓦂�Ă���B
					// �s�l�݃`�F�b�N
					// ���肪�Ȃ��ꍇ
					MovePicker<true> move_picker2(n);
					if (move_picker2.empty()) {
						auto& entry1 = transposition_table->LookUp<true>(n);
						entry1.lock();
#ifdef DFPN_MOD_V2
						if (entry1.pn != 0) {
#else
						if (true) {
#endif
							// TODO20231026: �㏑���̂�
							entry1.pn = kInfinitePnDn;
							entry1.dn = 0;
							// ���؋�
							// �����Ă��鎝������ő吔�ɂ���(���̎������������)
							entry1.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
						}

						n.undoMove(m2);    // TODO: undoMove() �d���Ȃ�N���e�B�J���Z�N�V��������ǂ����B
						entry.lock();
#ifdef DFPN_MOD_V2
						if (entry.pn != 0) {
#else
						if (true) {
#endif
							// TODO20231026: �㏑���̂�
							entry.pn = kInfinitePnDn;
							entry.dn = 0;
							// �q�ǖʂ̔��؋��ݒ�
							// �ł�Ȃ�΁A���؋��폜����
							if (m2.isDrop()) {
								entry.hand = entry1.hand;
								entry.hand.minusOne(m2.handPieceDropped());
							}
							// ���̋������Ȃ�΁A���؋�ɒǉ�����
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
						entry.unlock();    // TODO: ������unlock ���v���C������ȁB
						entry1.unlock();
						return;
					}
					n.undoMove(m2);
					goto NO_MATE;
				}

				n.undoMove(m2);
			}

			// ���ׂċl��
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
				// TODO20231026: �㏑���̂�
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
	//     : �ǂ����Ă��̃^�C�~���O�Ő���蔻�肷����H
	//       -> ���炭�₯�ǁAdfpn �Ăԋǖʂł͋l�݂̕��������������Ȃ����ȁB
	//          (��ɒZ�萔�̋l�݂��m�F���Ă����������A���̊֐��𔲂���Ɗ��҂���鎞�Ԃ͏������Ȃ�B)
	//     : ���������Ȃ�]���l�������������A
	//       �Ⴆ��depth=6��depth=9 �œ����ǖʂ��o�Ă����ꍇ�Adepth=6 �̋l�ݕs�l�݂������Ƃ��Ēl�������������邪�A
	//       qhapaq �̋L���̂悤�ɁA�����͓���������ԉ����ǖʂŏ��߂ĕs�l�݈������Ȃ���΂Ȃ�Ȃ��̂ŁA
	//       depth=6 �ł͕s�l�݂Ƃ��Ă͂����Ȃ�(�͂�)�B�u
	// �����̃`�F�b�N
	// TODO20231026: �㏑���̂�(�ȉ���switch ���S��)
	switch (n.isDraw(16)) {
	case RepetitionWin:
		// �A������̐����ɂ�鏟��
		if (or_node) {
			// �����͒ʂ�Ȃ��͂�
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
		// �A������̐����ɂ�镉��
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
			// �����͒ʂ�Ȃ��͂�
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
		// ���ʂ̐����
		// �����͒ʂ�Ȃ��͂�
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
			//     : AndNode �ŗD�z�ǖʂɂȂ����ꍇ�AAndNode ���_�ŗL���ɂȂ��Ă��ŁA
			//       OrNode ���_�ł͂��̋ǖʂɑJ�ڂ��ׂ��łȂ��B
			//       �]���āAOrNode ���_��"�}�C�i�X�̕]���l" �ƂȂ�悤��pn = inf �Ƃ���B
			// AND�m�[�h�ŗD�z�ǖʂɂȂ��Ă���ꍇ�A���O�ł���(OR�m�[�h�őI������Ȃ��Ȃ�)

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
	//     : �ȉ��̏�񂪂���΁A�u���\�ɒ��ڃA�N�Z�X(LookUpDirect)�o����B
	// �q�ǖʂ̃n�b�V���G���g�����L���b�V��
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
	// ����I��root �ɖ߂��Ă��āA�܂��ēx���g�̒S��depth �֌���̍ŗǎ菇�ŒH���Ă����A�̂��J��Ԃ��͂��B�B�B
	if (threadid && (game_ply - 1) && game_ply <= threadid) {    // ���C���X���b�h�ƃ��[�g�m�[�h��skip ���Ȃ�
		is_skip = true;
	}
	//if (threadid && (game_ply - 1)) {    // �_��
	//	//is_skip = ((game_ply + skipPhase[threadid]) / skipSize[threadid]) % 3;    // �_��
	//	is_skip = ((game_ply + skipPhase[threadid]) / skipSize[threadid]) % 2;
	//}
	//int node_count_down = (entry_num_searched / 10) + 1;    // �_��
	//int node_count_down = my_max(10LL - entry_num_searched, 1);    // �_��
	int node_count_down = 1;

	// NOTE
	//     : �C�ӂ�node �̒T������maxSearchNode �𒴂��Ă͂Ȃ�Ȃ��B
	while (searchedNode < maxSearchNode && !_should_stop<shared_stop>()) {
		//if (threadid == 23) {
		//	std::stringstream ss;
		//	ss << "W";
		//	enqueue_inf_debug_str(ss.str());
		//}

		// NOTE: ���炭num_searched �ɂ��đ���łȂ��X�V���Ȃ����̂͂��������B
		entry.lock();
		++entry.num_searched;    // TODO: ���ꂾ���ׂ̈�lock ����̂͂��������Ȃ���ȁB�ǂ����ړ��������B�B�B

		entry.unlock();

		Move best_move;
		// NOTE
		//     : �ċA���鎞�ɁAthpn, thdn �ɓn���l�ŁB
		//       �q�m�[�h��root �Ƃ���T���ɉe���B
		//     : �����͍Ō��set ����̂ŁA��xset ���ꂽ��͎���������I��const
		int thpn_child;
		int thdn_child;

		// NOTE
		//     : ���݌��Ă���ǖʂ̏��(=entry) ���A�q�m�[�h�̏��(=child_entry) �����ɍX�V
		//     : �T���o�H�����߂���I���܂ň�{�̒������ň�M�����o����̂ŁA
		//       PUCT �݂����ɂ킴�킴backup �Ƃ������Ƃ��A�q�m�[�h��pn, dn ���W�v���邾����
		//       ���݂�node ��pn, dn ���X�V�\���Ċ����H
		//       -> ����A���[��A�Ȃ񂩗��R�������ɈႤ�C������ȁB
		// expand and compute pn(n) and dn(n);
		if (or_node) {
			// OR�m�[�h�ł́A�ł��ؖ����������� = �ʂ̓������̌������Ȃ� = �l�܂��₷���m�[�h��I��
			int best_pn = kInfinitePnDn;
			int second_best_pn = kInfinitePnDn;    // NOTE: thpn_child �Ɏg���炵���B�B�B
			int best_dn = 0;    // NOTE: best_pn ���ŏ����L�^���邽�тɍX�V����̂ŁA
			//       pn, dn �̊Ԃɂ͖}�������Ɋ֌W�����藧�Ɖ��肵�Ă���͗l�B
			//       �܂�Abest_dn �ɂ́Adn �̍ő�l�̋ߎ��l���i�[�����͂��B
			uint32_t best_num_search = UINT32_MAX;    // NOTE: best_pn �Ȏq�m�[�h��num_searched 

#ifdef DFPN_MOD_V0
			int entry_pn = kInfinitePnDn;
			int entry_dn = 0;
#else
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
#endif
			// �q�ǖʂ̔��؋�̐ϏW��
			u32 pawn = UINT_MAX;
			u32 lance = UINT_MAX;
			u32 knight = UINT_MAX;
			u32 silver = UINT_MAX;
			u32 gold = UINT_MAX;
			u32 bishop = UINT_MAX;
			u32 rook = UINT_MAX;
			bool repeat = false; // �ő�萔�`�F�b�N�p    // TODO: ����Ȃ��˂�B
			//int tmpCount = 0;    // debug
			for (const auto& move : move_picker) {
				auto& ttkey = ttkeys[&move - move_picker.begin()];
				// TODO
				//     : ����lock ���邩�A���S�ȃR�s�[������Ă������g�����A�ǂ������������낤���B�e�X�g���ׂ����ȁ`�H
				//       �Ƃ������A�r���Ō��ʕς��̊��ƕs�s���ȋC������ȁB�m��񂯂ǁB
				//       _mtx �̓R�s�[���Ȃ��ėǂ�����Astruct TTEntryNoMtx ������āA�����ɃR�s�[�g�p������B
				//const auto& child_entry = transposition_table->LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				// TODO20231026: ���̎Q�Ƃ̂݁B
				auto& child_entry = transposition_table->LookUpDirect(
					*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1
				);    // �{����const auto&
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
				//     : �����ł́A"depth ���قȂ邪�A�D�z�֌W�𖞂�������Ԃ�"�A����������񊈂���B
				//       ���̂Ȃ�ALookUp ����entry ��pn, dn ��(�^��(pn == 0���l�݂Ƃ��Ĉ����Ă���))�g���Ă邩��ł���B
				if (child_entry.pn == 0) {    // NOTE: OrNode �Ŏq�m�[�h�̈���l�݂������̂ŁA���݂̃m�[�h���l��
					// �l�݂̏ꍇ
#ifdef DFPN_MOD_V0
					entry_pn = 0;
					entry_dn = kInfinitePnDn;
#else
					entry.pn = 0;
					entry.dn = kInfinitePnDn;
#endif
					// �q�ǖʂ̏ؖ����ݒ�
					// �ł�Ȃ�΁A�ؖ���ɒǉ�����
					if (move.move.isDrop()) {
						const HandPiece hp = move.move.handPieceDropped();    // NOTE: move �őł�����
						// NOTE
						//     : �ؖ���͏����������ǂ��̂ŁA�ł�����hp ���ؖ���ɉ����K�v�����ŏ�������B
						//       entry.hand (�����hp �̏ؖ���̐��̍ŏ�) vs entry.hand.plusOne(hp) (����Ŕ������� hp�̏ؖ���̐�) �Ŕ�r���āA��菬���������̗p�B
						//       ���̎��Ahp �����łȂ��A������S�̂��X�V���Ă��鎖�ɒ��ӁB
						//       (hp �̐�������ύX����ƁA���܂ł��w����A��z�肵���ؖ���ŁA����̂��w����B(=move)��z�肵���ؖ���̎��Ɉ�@�ł���B)
						// TODO: ����Ahp �����Ŕ��f�����ɁA�ؖ���̑����Ƃ��ŁA�ǂ����̗p���邩���߂������ǂ���ˁH
						entry.lock();
						if (entry.hand.numOf(hp) > child_entry.hand.numOf(hp)) {
							// TODO20231026
							//     : ��̏������ŎQ�Ƃ��Ă邪�A����͏㏑���݂̂ƌ�����񂶂�Ȃ����ȁB�B�B�H
							//       ���ėp�������߂��邩���m�F���Ă邾�����Ǝv���B��̏������͂ˁB
							entry.hand = child_entry.hand;
							entry.hand.plusOne(move.move.handPieceDropped());    // TODO: ���́A����A.plusOne(hp); �ɂ��Ȃ��H
						}
						// TODO: ����unlock�Aif else �S�̂ɂ��āA�X�ɍŌ��unlock ���Ȃ�����͂��B
						// (break �Ŕ������炷���ɂ܂�entry.lock() ����̂ŁB�B�B)
						entry.unlock();
					}
					// TODO
					//     : ������ꍇ�́A�ⓚ���p�ō���̎w����ɂ��Ă̏ؖ�����̗p������ۂ��B���́H�H
					// ���̋������Ȃ�΁A�ؖ����폜����
					else {
						const Piece to_pc = n.piece(move.move.to());
						if (to_pc != Empty) {    // NOTE: �������ł���
							entry.lock();
							// TODO20231026: �㏑���݂̂ƌ��������B�B�B�H
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
					//     : ����OrNode �A�܂�l�݂ɂ���Or �ł���A�l�݂����������̂�break
					child_entry.unlock();
					break;
				}
				// TODO
				//     : AndNode �̎��͂���ɑΉ����鏈���������́A�Ȃ����Ȃ����H
				//       
				// NOTE
				//     : root node(entry) ���s�l�݂Ȃ��{������ʂ�(�͂�)�B
				//       (�u���\���������ǖ�(entry)�ł͕s�l�݂ƌ����؂ꂽ���ǂ��̋ǖʂł͈Ⴄ�ꍇ(�q�m�[�h�ŋl��ł�ǖʃA��)�A���if �ɂ�)
				//     : �ȉ��̂悤�ɁA���؋�o�^����Ă�C�ӂ̎q�m�[�h�́A�C�ӂ̔��؋�(ex ��, ��, �j�n, ...) �ɂ��Ă̍ŏ��l�����߂�΁A���ꂪ�����ϏW���ł���B
				//     : true �ɂ���Ӗ��Ȃ��B���̂Ȃ�A���؋�͕s�l�݂̎��̂ݕK�v�ł���A�s�l�݂Ƃ͌����؂�Ȃ��Ɣ��������u�ԁA
				//       �킴�킴���؋���v�Z����Ӗ��������Ȃ�B
				//       -> if (true) ����if (entry_dn == 0) �ɁB
#ifdef DFPN_MOD_V0
				else if (entry_dn == 0) {
#else
				else if (entry.dn == 0) {
#endif
					if (child_entry.dn == 0) {
						const Hand& child_dp = child_entry.hand;
						// ��
						const u32 child_pawn = child_dp.exists<HPawn>();
						if (child_pawn < pawn) pawn = child_pawn;
						// ����
						const u32 child_lance = child_dp.exists<HLance>();
						if (child_lance < lance) lance = child_lance;
						// �j�n
						const u32 child_knight = child_dp.exists<HKnight>();
						if (child_knight < knight) knight = child_knight;
						// ��
						const u32 child_silver = child_dp.exists<HSilver>();
						if (child_silver < silver) silver = child_silver;
						// ��
						const u32 child_gold = child_dp.exists<HGold>();
						if (child_gold < gold) gold = child_gold;
						// �p
						const u32 child_bishop = child_dp.exists<HBishop>();
						if (child_bishop < bishop) bishop = child_bishop;
						// ���
						const u32 child_rook = child_dp.exists<HRook>();
						if (child_rook < rook) rook = child_rook;
					}
				}
				// NOTE
				//     : pn �͍ŏ��l��I�сAdn �͍��v�l�����
#ifdef DFPN_MOD_V0
				entry_pn = std::min(entry_pn, child_entry.pn);
				entry_dn += child_entry.dn;
#else
				entry.pn = std::min(entry.pn, child_entry.pn);
				entry.dn += child_entry.dn;
#endif

				// �ő�萔�ŕs�l�݂̋ǖʂ��D�z�֌W�Ŏg�p����Ȃ��悤�ɂ���
				if (child_entry.dn == 0 && child_entry.num_searched == REPEAT)
					repeat = true;

				// NOTE
				//     : �ؖ�������菬���� or
				//       �ؖ����������Ȃ�T��������菭�Ȃ�
				// TODO
				//     : ���́Achild_entry.num_searched �̒T���������Ȃ������ǂ��́H
				//       ����܂ł̒T���������Ȃ������A�����ؖ����ł����l�݂₷�������Ęb�H
				//       -> ���ɂ������Ƃ�����A�t�ɂ���܂ł̒T���ł͋l�݂��ؖ����₷���ǖʂ�T�����؂����̂ŁA
				//          ���ꂩ��̒T���ł͓���ǖʂ��c���Ă�����ĂȂ��Hdfpn �̐����I�ɂ��y�����ȓz����T�������₵���H
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
			}    // NOTE: for �I��

			// TODO
			//     : kInfinitePnDn ���ǂ�������ʂő������邩 �c���ł��ĂȂ��B
			//     : ����́A�q�m�[�h�ɕ���dn = kInfinitePnDn �ȓz���������ꍇ��clip ����̂��_���H
			entry.lock();
#ifdef DFPN_MOD_V0
#ifdef DFPN_MOD_V2
			if (entry.pn != 0 && entry.dn != 0) {    // �������ޒl�̗��������S�ɖ��m�̏ꍇ�A����pn, dn �ǂ�����`�F�b�N����B
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
				// TODO20231026: �㏑���̂�
				entry.pn = entry_pn;
				entry.dn = std::min(entry_dn, kInfinitePnDn);
			}
#else
			entry.dn = std::min(entry.dn, kInfinitePnDn);
#endif



			// NOTE
			//     : OrNode �͕s�l�݂Ɋւ���And �Ȃ̂ŁA�S�����Ă���łȂ���dn == 0 �͓r���ł͌����؂�Ȃ��B
			//       �S�����I������̂ŁA�����ŏ��߂�"dn == 0 �Ȃ�s�l�݂ł���" ���^�ƂȂ�B
			//     : df-pn �̓S���u�l�� || �s�l�� ������������A�ؖ��� || ���؋� ���v�Z���Ēu���\��set�v
			//     : ���̍��̋ǖʂ��s�l�݂ł���(entry_dn==0)���甽�؋��set ����̂ł���B
			//       ����������ɕύX����Ă��邩������Ȃ�entry.dn �̒l��p���Ă͂Ȃ�Ȃ��B
			if (entry_dn == 0) {
				// �s�l�݂̏ꍇ
				//cout << n.hand(n.turn()).value() << "," << entry.hand.value() << ",";
				// NOTE
				//     : bool repeat = false; �Ȃ�t���O�ɂ��ẴR�����g��A�ȉ��̃R�����g�Ȃǂ�����ɁA
				//       REPEAT �Ƃ͍ő�萔�ŒT�����f�������Ƃ������t���O�Ȃ̂��������ȁB
				// �ő�萔�ŕs�l�݂̋ǖʂ��D�z�֌W�Ŏg�p����Ȃ��悤�ɂ���
				if (repeat)
					entry.num_searched = REPEAT;
				else {
					// TODO20231026
					//     : else if (pawn > curr_pawn) pawn = curr_pawn; �ł͂Ȃ��񂾁H
					//       �����ł͂Ȃ��āA��������g�傷������ɍX�V����񂾁H�Ȃ�ŁH
					// ��肪�ꖇ�������Ă��Ȃ���ނ̐��̎�����𔽏؋��폜����
					u32 curr_pawn = entry.hand.template exists<HPawn>(); if (curr_pawn == 0) pawn = 0; else if (pawn < curr_pawn) pawn = curr_pawn;
					u32 curr_lance = entry.hand.template exists<HLance>(); if (curr_lance == 0) lance = 0; else if (lance < curr_lance) lance = curr_lance;
					u32 curr_knight = entry.hand.template exists<HKnight>(); if (curr_knight == 0) knight = 0; else if (knight < curr_knight) knight = curr_knight;
					u32 curr_silver = entry.hand.template exists<HSilver>(); if (curr_silver == 0) silver = 0; else if (silver < curr_silver) silver = curr_silver;
					u32 curr_gold = entry.hand.template exists<HGold>(); if (curr_gold == 0) gold = 0; else if (gold < curr_gold) gold = curr_gold;
					u32 curr_bishop = entry.hand.template exists<HBishop>(); if (curr_bishop == 0) bishop = 0; else if (bishop < curr_bishop) bishop = curr_bishop;
					u32 curr_rook = entry.hand.template exists<HRook>(); if (curr_rook == 0) rook = 0; else if (rook < curr_rook) rook = curr_rook;
					// TODO20231026
					//     : �����͉��H�Q�ƂƏ㏑�������H����Ƃ��㏑���̂݁H
					// ���؋�Ɏq�ǖʂ̏ؖ���̐ϏW����ݒ�
					entry.hand.set(pawn | lance | knight | silver | gold | bishop | rook);
					//cout << entry.hand.value() << endl;
				}
			}
			else {
				// NOTE
				//     : if (entry.pn >= thpn || entry.dn >= thdn) { break; } �ƕ]�����Ă���A
				//       �����̌Z��ǖʂ̃X�R�A�𒴉߂��ď��߂đł��؂肽���̂ŁA"+1" ����B
				if constexpr (EPS == 0) {    // �W��
					thpn_child = std::min(thpn, second_best_pn + 1);
				}
				else if constexpr (EPS > 0) {
					// ���q�m�[�h�̒����؍�
					thpn_child = std::min(thpn, static_cast<int>(_EPS_PLUS_ONE * second_best_pn + 1));
				}
				else {
					// ���q�m�[�h�ɒZ���؍�
					thpn_child = std::min(
						thpn, best_pn + static_cast<int>(_EPS_PLUS_ONE * (second_best_pn - best_pn) + 1)
					);
				}
				thdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn);
			}
			// TODO
			//     : �ꉞ�Aif (entry.dn == 0) �̏ꍇ�͍Ō��unlock() ���āA
			//       else �̏ꍇ��thdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn); 
			//       �������ɂ����unlock() ����΂�����Ƒ����Ȃ�͂��B�B�B
			//entry.unlock();    // NOTE: 臒l�m�F�̏I���Ɉړ�
		}
		else {    // AndNode
			// AND�m�[�h�ł͍ł����ؐ��̏����� = ����̊|�����̏��Ȃ� = �s�l�݂������₷���m�[�h��I��
			int best_dn = kInfinitePnDn;
			int second_best_dn = kInfinitePnDn;
			int best_pn = 0;
			uint32_t best_num_search = UINT_MAX;


			// NOTE: MOD_V0 ��K�p����ۂɁA������entry.pn = 0, entry.dn = kInfinitePnDn ���폜����̂�Y��Ă����B
#ifdef DFPN_MOD_V0
			int entry_pn = 0;
			int entry_dn = kInfinitePnDn;
#else
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
#endif

			// �q�ǖʂ̏ؖ���̘a�W��
			u32 pawn = 0;
			u32 lance = 0;
			u32 knight = 0;
			u32 silver = 0;
			u32 gold = 0;
			u32 bishop = 0;
			u32 rook = 0;
			bool all_mate = true;    // NOTE: ����܂łɏ��������q�m�[�h�̑S�Ă��l�݂ł������true
			for (const auto& move : move_picker) {
				auto& ttkey = ttkeys[&move - move_picker.begin()];
				//const auto& child_entry = transposition_table->LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1);
				// TODO20231026: ���̎Q�Ƃ̂݁B
				auto& child_entry = transposition_table->LookUpDirect(
					*ttkey.entries, ttkey.hash_high, ttkey.hand, n.gamePly() + 1
				);    // �{����const auto&
				child_entry.lock();

				if (all_mate) {
					if (child_entry.pn == 0) {
						// NOTE: �ؖ���̘a�W�������߂�B
						const Hand& child_pp = child_entry.hand;
						// ��
						const u32 child_pawn = child_pp.exists<HPawn>();
						if (child_pawn > pawn) pawn = child_pawn;
						// ����
						const u32 child_lance = child_pp.exists<HLance>();
						if (child_lance > lance) lance = child_lance;
						// �j�n
						const u32 child_knight = child_pp.exists<HKnight>();
						if (child_knight > knight) knight = child_knight;
						// ��
						const u32 child_silver = child_pp.exists<HSilver>();
						if (child_silver > silver) silver = child_silver;
						// ��
						const u32 child_gold = child_pp.exists<HGold>();
						if (child_gold > gold) gold = child_gold;
						// �p
						const u32 child_bishop = child_pp.exists<HBishop>();
						if (child_bishop > bishop) bishop = child_bishop;
						// ���
						const u32 child_rook = child_pp.exists<HRook>();
						if (child_rook > rook) rook = child_rook;
					}
					// NOTE: �s�l�݂����������̂ŁAall_mate ��false �ƂȂ�B
					else {
						all_mate = false;
					}
				}
				if (child_entry.dn == 0) {
					// �s�l�݂̏ꍇ
#ifdef DFPN_MOD_V0
					entry_pn = kInfinitePnDn;
					entry_dn = 0;
#else
					entry.pn = kInfinitePnDn;
					entry.dn = 0;
#endif
					entry.lock();
					// �ő�萔�ŕs�l�݂̋ǖʂ��D�z�֌W�Ŏg�p����Ȃ��悤�ɂ���
					if (child_entry.num_searched == REPEAT)
						entry.num_searched = REPEAT;
					else {
						// �q�ǖʂ̔��؋��ݒ�
						// �ł�Ȃ�΁A���؋��폜����
						if (move.move.isDrop()) {
							const HandPiece hp = move.move.handPieceDropped();
							// TODO20231026: ���炭�㏑���݂̂ƌ�����͂��B
							if (entry.hand.numOf(hp) < child_entry.hand.numOf(hp)) {
								entry.hand = child_entry.hand;
								entry.hand.minusOne(hp);
							}
						}
						// ���̋������Ȃ�΁A���؋�ɒǉ�����
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
					// TDOO: entry ��break �������璼��ɖ�lock() ���邩��A�����Ŏ�����K�v�͂Ȃ��B
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
			}    // NOTE: for �I��
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
				// TODO20231026: �㏑���̂݁B
				entry.pn = std::min(entry_pn, kInfinitePnDn);
				entry.dn = entry_dn;
			}
#else
			entry.pn = std::min(entry.pn, kInfinitePnDn);
#endif

			// NOTE
			//     : �����ŏ��߂�"pn == 0 �Ȃ�΋l�݂ł���" ���^�ƂȂ�B
			//     : df-pn �̓S�����A���؋�̌v�Z��set
			if (entry_pn == 0) {
				// �l�݂̏ꍇ
				//cout << n.toSFEN() << " and" << endl;
				//cout << bitset<32>(entry.hand.value()) << endl;
				// �ؖ���Ɏq�ǖʂ̏ؖ���̘a�W����ݒ�
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
				//     : �����͉��H�Q�ƂƏ㏑�������H����Ƃ��㏑���̂݁H
				// NOTE
				//     : �i���[�~�ł�����Ƃ���
				//     : ������AndNode �Ȃ̂ŁAsetPP() ����Ƃ��́A�󂯑����ꖇ�������ĂȂ��āA�U�ߎ肪�Ɛ肵�Ă���
				//       �ؖ���ɒǉ����Ȃ��Ƃ����Ȃ��B�܂�Athem ��oppositeColor ��hand ��n���ׂ��ł́H
				// ��肪�ꖇ�������Ă��Ȃ���ނ̐��̎�������ؖ���ɐݒ肷��
				if (!(n.checkersBB() & n.attacksFrom<King>(n.kingSquare(n.turn())) || n.checkersBB() & n.attacksFrom<Knight>(n.turn(), n.kingSquare(n.turn())))) {
					entry.hand.setPP(n.hand(oppositeColor(n.turn())), n.hand(n.turn()));
				}
			}
			else {
				// TODO
				//     : ���̂悤�ɍX�V����Ӗ��͉��H�H�H�H
				//       -> �O���OrNode �ɂ�����őP&���P�̌Z��ǖʂɂ��Ă�pn�̍�����PnPrev_12�A
				//          �����OrNode �ɂ�����őP&���P�̌Z��ǖʂɂ��Ă�pn�̍�����PnNext_12 �Ƃ��A
				//          �����OrNode �ɂ�����őP�̋ǖ�(�q�m�[�h, �w����) ��pn ��Pn_1 �Ƃ���ƁA
				//          �����OrNode �ɂ����āAthpn_child = Pn_1 + min(��PnPrev_12, ��PnNext_12) + 1 �Ƃ��邱�Ƃɓ������B
				//          ���ɁAmin(��PnPrev_12, ��PnNext_12) == ��PnNext_12 �̏ꍇ�Athpn_child = second_best_pn + 1 �Ƃ��邱�Ƃɓ������B
				thpn_child = std::min(thpn - entry.pn + best_pn, kInfinitePnDn);
				//thdn_child = std::min(thdn, second_best_dn + 1);
				if constexpr (EPS == 0) {
					thdn_child = std::min(thdn, second_best_dn + 1);
				}
				else if constexpr (EPS > 0) {
					// ���q�m�[�h�̒����؍�
					thdn_child = std::min(thdn, static_cast<int>(_EPS_PLUS_ONE * second_best_dn + 1));
				}
				else {
					// ���q�m�[�h�ɒZ���؍�
					thdn_child = std::min(
						thdn, best_dn + static_cast<int>(_EPS_PLUS_ONE * (second_best_dn - best_dn) + 1)
					);
				}
			}
			//entry.unlock();    // NOTE: 臒l�m�F�̏I���Ɉړ�
		}
		// NOTE
		//     : �l��(pn == 0) �ƕ��������ꍇ�́Adn == inf �ƂȂ��Ă���̂ŁAdn �̐�����break ����B
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
			//     : �[���ɉ����Ă���ς���Ƃ��A����������3�Ƃ�4�ɂ���Ƃ��ˁH�F�X����B
			thpn_child = std::min(static_cast<int>(thpn_child * THPN_MULTIPLY), kInfinitePnDn);
			thdn_child = std::min(static_cast<int>(thdn_child * THDN_MULTIPLY), kInfinitePnDn);
		}
		dfpn_inner<!or_node, shared_stop>(n, thpn_child, thdn_child/*, inc_flag*/, maxDepth, searchedNode, threadid);
		n.undoMove(best_move);

		node_count_down -= is_skip;
		// NOTE
		//     : �����Ŕ����Ă͂Ȃ�Ȃ��͂��B������̂́Anode�����Ŋ��S�ɒT���𒆎~���鎞�̂݁B
		//       ��΂ɁA���̃m�[�h�ɁA�q�m�[�h����A�҂��Ă������́A����whlie�߂̑啔�����߂�Apn, dn �̍X�V��Ƃ��K�v�B
		//       ���̊֐������œ������Ă���肪�����̂́A�K��pn, dn �̍X�V�����ɏオ��ɍs���Ă��邩��ł���B
		//       ��ł��X�V���ʂ����ƁA���̏u�Ԕ��f����Ȃ��q�m�[�h�ł�pn, dn �̍X�V ���o�Ă��Ă��܂��B
	}
}

// �l�݂̎�Ԃ�
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

// FIXME: �i���ɏz���ďI���Ȃ����Ƃ�������ۂ��B�B�B�H
template<bool or_node, bool safe>
int ParallelDfPn::get_pv_inner(Position & pos, std::vector<Move>&pv) {
	std::stringstream ss;
	pos.print(ss);
	if (or_node) {
		// OR�m�[�h�ŋl�݂����������炻�̎��I��
		MovePicker<true> move_picker(pos);
		for (const auto& move : move_picker) {
			// NOTE: �R�C�c��lock, unlock �ȊO��const
			auto& child_entry = transposition_table->LookUpChildEntry<true>(pos, move);
			if (safe) child_entry.lock();

#ifdef DEBUG_GETPV_20231026_0
			const auto debug_bk = pos.getBoardKey();
			// debug
			// ����ǖʂł́A����q�ǖʈȊO�ɂ͑J�ڂ����A����q�ǖʈȊO�͑S�Ĕ�΂��B
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
					// NOTE: �ȉ��̂ǂ��炩�ł���΍ċA�B
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
		// AND�m�[�h�ł͋l�݂܂ł��ő�萔�ƂȂ���I��
		int max_depth = 0;
		std::vector<Move> max_pv;    // NOTE: �ő��PV
		MovePicker<false> move_picker(pos);
		for (const auto& move : move_picker) {
			// NOTE: �R�C�c��lock, unlock �ȊO��const
			auto& child_entry = transposition_table->LookUpChildEntry<false>(pos, move);
			if (safe) child_entry.lock();

#ifdef DEBUG_GETPV_20231026_0
			const auto debug_bk = pos.getBoardKey();
			// debug
			// ����ǖʂł́A����q�ǖʈȊO�ɂ͑J�ڂ����A����q�ǖʈȊO�͑S�Ĕ�΂��B
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
			// ����w����ɂ���đJ�ڂ���ǖʂ̏���\������B(pn == 0 �Ȃ�A���̋ǖʎ��͖̂��Ȃ��Ƃ������ƂɂȂ�B)
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
			//     : .dn == 0 �̏ꍇ���l���Ȃ��B���̂Ȃ炱���ɗ���Ƃ��ɒʂ����m�[�h��pn == 0�ł���A
			//       ������AndNode �Ȃ̂ŁA�A���S���Y������������΂����͓��R�S�Ă̎w���肪�l�ނ͂��ŁA
			//       ����Ȃ��̃`�F�b�N���Ȃ��B
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
						// 1��l�݃`�F�b�N
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

				// NOTE: ��蒷��PV��������΍X�V
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
//     : safe �Ȃ�A���̒T���Ɣ���Ă����Ȃ��B
//     : ���̊֐����̂����ŌĂԂ��Ƃ͑z�肵�Ă��Ȃ����A���Ȃ������ȋC�͂��Ă�B
// PV�Ƌl�݂̎�Ԃ�
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

// �l�����T���̃G���g���|�C���g
template<bool shared_stop>
bool ParallelDfPn::dfpn(Position & r, int64_t & searched_node, const int threadid) {
	// �L���b�V���̐����i�߂�

	//std::cout << "debug: DLSHOGI dfpn::dfpn" << std::endl;
	//std::cout << "info: " << get_option_str() << std::endl;

	searched_node = 0;    // NOTE: �T���m�[�h����reset
	if (!r.inCheck()) {
		// 1��l�݃`�F�b�N
		Move mate1ply = r.mateMoveIn1Ply<mmin1ply_Additional>();
		if (mate1ply) {
			auto& child_entry = transposition_table->LookUpChildEntry<true>(r, mate1ply);
			child_entry.lock();
#ifdef DFPN_MOD_V3
			if (child_entry.dn != 0) {    // ���΂ɑ��v���Ǝv�����ǂˁB
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

// �l�����T���̃G���g���|�C���g
template<bool shared_stop>
bool ParallelDfPn::dfpn_andnode(Position & r, int64_t & searched_node, const int threadid) {
	// ���ʂɉ��肪�������Ă��邱��

	// �L���b�V���̐����i�߂�
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
//         : .h �ɐ錾�A.cpp �ɒ�`�������Ă���ꍇ�A�����I�C���X�^���X�����K�v�B
//         :  �����I�C���X�^���X�������Ȃ��ꍇ�A�����R���p�C���������ňȉ��̎��ۂ���������B
//            .h ��include ���Ă�t�@�C���ł́A�錾������̂ň�U�R���p�C�����ʂ�A
//            .cpp �ł́Atemplate �̊֐����g���Ă��Ȃ��̂ŃC���X�^���X�����ꂸ�ɃR���p�C�������B
//            �ȏ���A.obj ��link ���鎞�ɁA�ǂ��ɂ���`�������I�Ƃ������̂��������ALNK2001 �ƂȂ�B
//         : �����I�C���X�^���X���́Atemplate �Ə����Btemplate<> �ł͖����B
template std::tuple<std::string, int, Move> ParallelDfPn::get_pv<true>(Position & pos);
template std::tuple<std::string, int, Move> ParallelDfPn::get_pv<false>(Position & pos);

template void ParallelDfPn::print_entry_info<true>(Position & n);
template void ParallelDfPn::print_entry_info<false>(Position & n);

template bool ParallelDfPn::dfpn<true>(Position& r, int64_t& searched_node, const int threadid);
template bool ParallelDfPn::dfpn<false>(Position& r, int64_t& searched_node, const int threadid);

template bool ParallelDfPn::dfpn_andnode<true>(Position& r, int64_t& searched_node, const int threadid);
template bool ParallelDfPn::dfpn_andnode<false>(Position& r, int64_t& searched_node, const int threadid);

#endif    // DFPN_PARALLEL4