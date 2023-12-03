#pragma once

#include <algorithm>
#include <vector>
#include <stdio.h>
#include "cshogi.h"
#include "types.hpp"
#include "util.hpp"
#include "random_player.hpp"

// TODO
//     : �����G���W���Ԃł̑΋ǁB
//       �Ⴆ�΁Amodel path �𕡐��w�肵����A�w�肵���������G���W���𗧂��グ�āA
//       ��������`���Ő������Ă����B(�ΐ�҂�random �őI�ԂƎ������N���ȁB)
//     : ���t�̈��k�BHuffmanCodedPosAndEval �g���̂͂���B�݊���������Ɗ���������ˁB

// HACK: ����A�N���X�̈Ӗ�������܂�Ȃ��B
class GensfenRandom {
public:
	GensfenRandom(
		unsigned long long n_gensfens_total,
		unsigned long long save_interval,
		int max_moves_to_draw,
		std::string file_name
	) : _n_gensfens_total(n_gensfens_total), _save_interval(save_interval),
		_max_moves_to_draw(max_moves_to_draw), _file_name(file_name)
	{
		_curr = 0;
		_player = RandomPlayer();
		_board = __Board();
		_board.reset();
	}

	void gen() {
		std::cout << get_now_str() << " : gensfen start" << std::endl << std::flush;
		int n_loops = _n_gensfens_total / _save_interval;
		for (int i = 0; i < n_loops; ++i) {
			std::ostringstream oss;
			oss << "[" << i + 1 << "/" << n_loops << "] gensfen";
			_gen(oss.str(), _save_interval);
		}
		std::cout << "info: gensfenrand done" << std::endl;
	}

private:
	RandomPlayer _player;
	__Board _board;
	int _curr;    // ���݂̋ǖʐ��B
				  // ���ɋǖʂ��i�[���ׂ�idx �ł�����B (�̂ɁA��ɋ���ۂ̉ӏ����w���B)
	unsigned long long _n_gensfens_total;
	unsigned long long _save_interval;
	int _max_moves_to_draw;
	std::string _file_name;
	//bool _random_name;

	// TODO
	//     : ply��ȏ�̎��ɂ̂ݏ����o��
	//     : ������s�B��̋��L����queue�Ȃ�vector �Ȃ�̔r�����b�N���e�X���b�h������I�Ɋl�����āApush_back() ����B
	//     : ���������̔��� & ��Ɉ��������ɂȂ�o�O�𒲍��B
	//       �e�X���b�h�̔r�����b�N���l�����銴�o�́A�����Ő�������Ɨǂ������ˁB�e�X���b�h������ƕʁX�̃^�C�~���O�ŃA�N�Z�X����悤�ɂȂ�͂��B
	// n_gensfens �ɂƂĂ��������l�����邱�Ƃ͂܂��Ȃ����낤�A�Ɖ��肵�������ł̎����B
	// @arg n_gensfens
	//     : �e�΋ǏI�����ɑ������ǖʐ����m�F���āA����𒴂��Ă�����I���B
	//       class �{�̂Ƃ͈Ⴂ�A���񐶐�����ǖʐ��ł��邱�Ƃɒ��ӁB
	// @arg max_moves_to_draw
	//     : max_moves_to_draw ���������ƂȂ�ő�萔�B
	void _gen(std::string proc_name, unsigned long long n_gensfens) {
		_curr = 0;
		std::vector<teacher_t> teachers(n_gensfens + _max_moves_to_draw - 1);
		Progress prog(proc_name, n_gensfens, 1e5);
		while (_curr < n_gensfens) {
			_board.reset();
			float result = 0.5;
			int start_idx = _curr;
			int ply = 0;
			if (!_player._isready()) {
				std::cout << "Error: isready failed" << std::endl;
			}
			_player._usinewgame();

			while (ply < _max_moves_to_draw) {    // ���݂܂ł̎萔��_max_moves_to_draw �����ł���΁A�����������
				// check
				// NOTE
				//     : _player._go() �����O�Ɉȉ���check ���K�v�B_go() �����������O��Ŏ������Ă���̂ŁB
				//     : �����ł͂��邪�A�ȉ���if ����result �́A�Ō�ɒ��肵����ԂƂ͔��΂̎�Ԃ��猩�����ʂł���B
				if (_board.is_game_over()) {
					result = 0;
					break;
				}
				else if (_board.is_nyugyoku()) {
					result = 1;
					break;
				}

				// TODO; �����ŁA�]���l�ɂ���Ă͓���������B
				// think
				int move = _player._go();

				// store
				// result �͎�����ɓZ�߂ď���
				ply = (_curr - start_idx) + 1;
				teachers[_curr].sfen = _board.toSFEN();    // �������Arandom player�̎v�l���Ԃ�2,3�{�|�����Ă�B
				teachers[_curr].move = move;
				teachers[_curr].ply = ply;
				teachers[_curr].value = 0.5;    // dummy

				// next
				++_curr;
				prog.step();
				_board.push(move);
				_player.direct_push(move);

				if (ply == _max_moves_to_draw) {    // �ő�萔���B�ɂ���������
					// HACK; max_moves_to_draw��ڂ����� or ���ʐ錾�łȂ��������_�ň��������Ƃ���B
					//if (board.is_game_over()) { result = -1; }
					//else if (board.is_nyugyoku()) { result = 1; }
					//else { result = 0; }
					result = 0.5;
					break;
				}
			}

			for (int i = _curr - 1; i >= start_idx; --i) {
				result = 1 - result;    // ���]�B(0.5 �Ȃ�0.5 �̂܂܁B)
				teachers[i].result = result;
			}
		}
		prog.finalize();
		// TODO: �����ŋǖʂ�shuffle ���������B
		// write .bin
		teachers.resize(n_gensfens);    // ������؂藎�Ƃ�

		write_teacher(teachers, _file_name, my_min(n_gensfens / 10, 1e6), true, true);
	}
};