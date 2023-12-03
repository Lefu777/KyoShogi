#pragma once

#include "cshogi.h"
#include "util.hpp"
#include <vector>
#include <memory>

#if defined(RANDOM_PLAYER) || defined(UCT_RANDOM_PLAYER) || defined(PUCT_PLAYER)

// TODO: static
constexpr int VALUE_NONE = -10000;

// HACK: struct �ł������H
class UctNode {
private:
	// ���̎��X�ɂ����ēK�؂�legal_moves_size ��set ����Ă���O��
	// �S�Ă�unique_ptr ����x�ɂ܂Ƃ߂�make_unique ����B
	void _make_unique_at_once() {
		child_nodes = std::make_unique<std::unique_ptr<UctNode>[]>(legal_moves_size);
		child_moves = std::make_unique<Move[]>(legal_moves_size);
		child_move_counts = std::make_unique<int[]>(legal_moves_size);
		child_sum_values = std::make_unique<float[]>(legal_moves_size);
		child_probabilitys = std::make_unique<float[]>(legal_moves_size);
	}

	void _init_single_node(int idx, Move move) {
		child_nodes[idx] = nullptr;    // �{���ɕK�v�ɂȂ�܂Ŏ��͎̂����Ȃ��B
		child_moves[idx] = move;
		child_move_counts[idx] = 0;
		child_sum_values[idx] = 0;
		child_probabilitys[idx] = 1;
	}

	bool _evaled;                // ���̋ǖʂɂ��Đ��_������true �ɁB

public:
	// ���݂����u�Ԃ���g�p�\�B
	int move_count;             // �m�[�h�̖K���    (child_move_counts �̍��v)
	float sum_value;            // �����̍��v    (child_sum_values �̍��v)
	float value;                // ���l

	// �W�J��Ɏg�p�\�B
	int legal_moves_size;    // HACK: child_num �̕����K�؂����ˁB

	// self �̃f�X�g���N�^���Ă΂��ƁA�ȉ���4��unique_ptr �ɑ΂��Ă��f�X�g���N�^���Ă΂�銴�����ȁH
	std::unique_ptr<std::unique_ptr<UctNode>[]> child_nodes;        // �q�m�[�h�B

	// TODO: ������͐�΂�atomic �ɂ���K�v������
	// HACK: ����4�A��̍\���̂ɓZ�߂������悢�����ˁBChildInfo �Ƃ��ɂ��B
	std::unique_ptr<Move[]> child_moves;            // �q�m�[�h�ւ̎w����(�G�b�W)
	std::unique_ptr<int[]> child_move_counts;      // �q�m�[�h�ւ̖K���
	std::unique_ptr<float[]> child_sum_values;     // �q�m�[�h�ɑJ�ڂ����ۂ�"�������̍��v"
	                                               // update_result() �ɂ��X�V�����B
	std::unique_ptr<float[]> child_probabilitys;   // �q�m�[�h�ւ̑J�ڊm�� by NN


	UctNode() {
		move_count = 0;
		sum_value = 0;
		value = VALUE_NONE;
		legal_moves_size = -1;    // 0�ȏ�̎��A�m�[�h���W�J����Ă���B
		_evaled = false;
	}

	~UctNode() {
		// HACK
		//     : ���炭���̊��ʂ͂Ȃ��Ă��A���̃f�X�g���N�^�̃X�R�[�v�����鎞��GC����邩��v���͂��B
		//       -> ���炭�Ƃ������A���[�J���ϐ������瓖����O�B�܂��ł��O�̈� of �O�ׂ̈ň�U���u�B
		{
			auto temp1 = std::move(child_nodes);
			auto temp2 = std::move(child_moves);
			auto temp3 = std::move(child_move_counts);
			auto temp4 = std::move(child_sum_values);
			auto temp5 = std::move(child_probabilitys);
		}
	}

	void set_evaled() { _evaled = true; }
	bool is_evaled() const { return _evaled; }

	// NOTE
	//     : ���̎w����(���̃m�[�h�ɂ�����w����)���c�_�ɋ�����܂łɓW�J����Ă�K�v������B
	// �w���肪0 �̏ꍇ�͕ʂŌ��o���āA�����܂ŗ��Ȃ��悤�ɂ���(����)�B
	// node �̓W�J
	// (�ЂƂ܂��A���̂悤�Ȏw���肪����Ƃ������Ƃ����ׂĔc�����邪�A���͕̂K�v�ɂȂ�܂ō쐬���Ȃ��B)
	void expand_node(const __Board& board) {
		__LegalMoveList lm(board);
		legal_moves_size = lm.size();
		_make_unique_at_once();
		for (int i = 0; i < lm.size(); lm.next(), ++i) {
			auto lm_move = lm.Move();
			_init_single_node(i, lm_move);
		}
	}

	// �q�m�[�h�̎��͍̂쐬���Ȃ��B
	// expand_node �́Alegal_move_size = 1 �œ���̎w���肾���W�J����version �Ƃ�������B
	void init_and_create_single_child_node(Move move) {
		legal_moves_size = 1;
		_make_unique_at_once();
		_init_single_node(0, move);
	}

	// �q�m�[�h�̎��̍쐬�B
	// expand_node() ���Ă΂ꂽ��ɂ̂ݎ��s�B
	UctNode* create_child_node(int idx) {
#ifdef DEBUG
		if (!(legal_moves_size >= 0)) {    // ���ɓW�J����Ă��鎞�B�T�����͂�����ɂȂ�͂��B
			std::cout << "Error: at create_child_node()" << std::endl;
			exit(1);
		}
#endif
		child_nodes[idx] = std::make_unique<UctNode>();
		// unique_ptr.get()����̂Ȃ񂩂��������ǂȁB
		// �܂��AGC����̂́A�������p�����ɂȂ��Ă���₩����͖����B
		return child_nodes[idx].get();
	}
	// @arg move: ���̎w����ɑΉ�����q�m�[�h�ȊO��S�ĉ������B
	UctNode* release_children_except_one(Move move) {
		if (legal_moves_size >= 0) {
			bool found = false;
			// HACK: �Q�Ɠn���������������B
			for (int i = 0; i < legal_moves_size; ++i) {
				if (child_moves[i] == move) {    // �����B
					found = true;
					if (!child_nodes[i]) {    // ���̂�������΍��B
						//std::cout << "info string rceo(): meet nullptr start" << std::endl;
						child_nodes[i] = std::make_unique<UctNode>();
						//std::cout << "info string rceo(): meet nullptr done" << std::endl;
					}
					else {
#ifdef DEBUG_
						std::cout << "info string rceo(): exist" << std::endl;
						std::cout << "info string rceo(): child_move_counts[i] = " << child_move_counts[i] << std::endl;
						std::cout << "info string rceo(): move_count = " << move_count << std::endl;
#endif
#ifdef DEBUG
						std::cout << "info string reuse subtree ["
							<< child_move_counts[i] << " / " << move_count << "]" << std::endl;
#endif
					}
					// else { // ���̂����łɗL��Ή��������ɂ�����ė��p�B }

					// �c��node ��0�ԖڂɎ����Ă���
					if (i != 0) {    // 0�ԖڂłȂ���
						// NOTE
						//     : ���̗v�f�͓���ւ��ĂȂ����ǁA�g�����
						//    -> �セ�������A0�Ԗڂ̃|�C���^�͉���ς݂Ȃ̂ŁA���̃|�C���^�͕ێ����Ȃ��Ă��ǂ��B
						child_nodes[0] = std::move(child_nodes[i]);
						child_moves[0] = child_moves[i];
						child_move_counts[0] = child_move_counts[i];
						child_sum_values[0] = child_sum_values[i];
						child_probabilitys[0] = child_probabilitys[i];
					}
				}
				else {
					//std::cout << "info string rceo(): got = " << __move_to_usi(child_moves[i]) << std::endl;
					// TODO: if (child_nodes[i]) �ɁB
					if (child_nodes[i]) {
						{
							// GC
							auto temp = std::move(child_nodes[i]);
						}
						// TODO: ��thread �œ���GC
					}
				}
			}
			if (found) {
				legal_moves_size = 1;
				return child_nodes[0].get();
			}
			else {
#ifdef DEBUG_
				std::cout << "info string not found subtree, move = " << __move_to_usi(move.value()) << std::endl;
#endif
				init_and_create_single_child_node(move);
				// ���̂��쐬���āA�����Ԃ��B
				// HACK: return create_child_node(0);
				return (child_nodes[0] = std::make_unique<UctNode>()).get();
			}
		}
		else {
#ifdef DEBUG_
			std::cout << "info string not found subtree, because this is unexpanded node, move = " << __move_to_usi(move.value()) << std::endl;
#endif
			init_and_create_single_child_node(move);
			// ���̂��쐬���āA�����Ԃ��B
			return (child_nodes[0] = std::make_unique<UctNode>()).get();
		}
	}

	Move get_move(const int idx) const {
		return child_moves[idx];
	}

	int get_move_count(const int idx) const {
		return child_move_counts[idx];
	}

	float get_winrate(const int idx) const {
		return child_sum_values[idx] / child_move_counts[idx];
	}

	int get_bestmove_idx() const {
		// NOTE: argmax_idx() ��const ObjTy& obj �ŎQ�Ƃ��Ă邾���Ȃ̂�ok�B�ړ�������Z�q�͎g���ĂȂ��B(�͂�)
		return argmax_idx(child_move_counts, legal_moves_size);
	}

	Move get_bestmove() const {
		return child_moves[this->get_bestmove_idx()];
	}

	void print_child_move_counts() const {
		for (int i = 0; i < legal_moves_size; ++i) {
			std::cout << "[" << i << "] " << child_move_counts[i] << std::endl;
		}
	}

	// �K��񐔂��������ɕ��ׂ�ꂽ�A�m�[�h�̃C���f�b�N�X���X�g��Ԃ�
    // �y���͖����̂ŁA�p�ɂɌĂяo���Ȃ�����
	auto get_sorted_idx_list() const {
		auto&& ret = sorted_idx_descending(child_move_counts, legal_moves_size);
		return ret;
	}

};

class UctTree {
private:
	std::unique_ptr<UctNode> _gamebegin_node;
	Key _starting_pos_key;

	// @arg node: node.reset() ������ɁAcurrent_head = node.get() ���邱�ƂƓ���
	void _reset_and_set_to_current(std::unique_ptr<UctNode>& node) {
		if (node) {
			auto temp = std::move(node);
		}
		node = std::make_unique<UctNode>();
		current_head = node.get();
	}

public:
	// NOTE
	//     : �|�C���^�Ȃ̂Œl���M���̂ŁA"private �ɂ����āAgetter �����p�ӂ��āA
	//       �O������͒l������邯�ǁA�M��Ȃ�" �݂����Ȃ��Ƃ��o���Ȃ��B
	UctNode* current_head;

	UctTree() {
		_starting_pos_key = 0;
		_reset_and_set_to_current(_gamebegin_node);
	}

	// TODO
	//     : �s�v�ɂȂ���Tree �̃f�X�g���N�g
	//     : �\�Ȃ�Εʂ�thread ��GC
	//     : �m�[�h�̍ė��p�B
	// �񐄏�
	void reset() {
		_reset_and_set_to_current(_gamebegin_node);
		std::cout << "info string Warning: reset() is deprecated." << std::endl;
	}

	bool reset_to_position(Key new_starting_pos_key, std::vector<Move> moves) {
		if (_starting_pos_key != new_starting_pos_key) {
			_reset_and_set_to_current(_gamebegin_node);
			std::cout << "info string got different starting pos" << std::endl;
		}

		_starting_pos_key = new_starting_pos_key;

		UctNode* old_head = current_head;
		UctNode* prev_head = nullptr;    // ������H��ۂɁA���݂̈��O��node ���w���B
		current_head = _gamebegin_node.get();    // �J�n�n�_����w���i�߂Ă����B
		bool seen_old_head = old_head == current_head;

		// ������H��
		for (const auto& move : moves) {
			//std::cout << "info string reset_to_position(): move = " << __move_to_usi(move.value()) << std::endl;
			prev_head = current_head;
			current_head = current_head->release_children_except_one(move);
			if (old_head == current_head) {
				seen_old_head = true;
			}
		}

		// TODO: ������Ȃ�������
		// ���̎��_�ŁAcurrent_head �́A�V����set ���ꂽ�ǖʂɑ�������node ���w�������B
		// old_head �������炸�A�����ǖʂ�set ���ꂽ��ł��Ȃ����A�ė��p�s��
		if (!seen_old_head && current_head != old_head) {
			if (prev_head != nullptr) {    // �J�n�ǖʂł͂Ȃ�
				// �ȑO�ɒT�������ǖʂ̑c��ł���\��������B
				// ���̎��Acurrent_head �����i�񂾐�ŁAold_head ������\��������B
				// �Ȃ̂ŁA�������Z�߂ĉ������B
				// TODO
				//     : 2, 3����old_head ������Ȃ�A�b�������炻��node �ɒH�蒅���͂��ŁA
				//       �������]�T������Ȃ�c���Ă������Ǝv�����ǁA�ʏ�̎��H����A�������������̂ڂ��āA
				//       ���r���[�ȋǖʂ�set ����邱�Ƃ͂Ȃ�����A�܂���U����ŁB
				_reset_and_set_to_current(prev_head->child_nodes[0]);
			}
			else {
				_reset_and_set_to_current(_gamebegin_node);
			}
		}
		return seen_old_head;
	}

};

// probabilitys �Ɋi�[���ꂽ���W�b�g���A�m���ɕϊ��B
inline void softmax_with_temperature(UctNode* node, const float temp) {
	// 0 �ɂ��Ă������ƂŁA0�ȏ�̑傫������������Ȃ����A
	// overflow �΍�͔������Ȃ��B
	float max_logit = 0;

	// ���x
	for (int i = 0; i < node->legal_moves_size; ++i) {
		float& logit = node->child_probabilitys[i];
		logit *= temp;
		if (logit > max_logit) {
			max_logit = logit;
		}
	}

	// �I�[�o�[�t���[�΍�ōő�l������
	float sum = 0;
	for (int i = 0; i < node->legal_moves_size; ++i) {
		float& logit = node->child_probabilitys[i];

		logit = expf(logit - max_logit);
		sum += logit;
	}

	// ���K��(softmax)
	for (int i = 0; i < node->legal_moves_size; ++i) {
		float& logit = node->child_probabilitys[i];

		logit /= sum;
	}
}

// HACK: �����ɒu���̂͐��������B�Ⴆ�΁A���t�����̎��ɁA���global_tree �������������ĂȂ���������B���ɂ����낢��A�e�Ɋp��������ˁB
extern std::unique_ptr<UctTree> global_tree;

#endif