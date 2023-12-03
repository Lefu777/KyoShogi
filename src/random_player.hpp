#pragma once

#include <iostream>
#include <ctime>
#include <cstdlib>

#include "base_player.hpp"
#include "MT.hpp"

// TODO
//     : ���ۃN���X�Ƃ������A���N���X�����B
//       �ق�ł����āAoverride �I�Ȃ��Ƃ��āA����Player�Ƃ��ăI�u�W�F�N�g�͎����Ă邯�ǁA
//       random_player �̂��Ăяo���A�I�Ȃ̂��������B
//     : �Ȃ񂩁Aheader �̒��g�ς����������ƁA.exe �ɔ��f�����B

class RandomPlayer : public BasePlayer {
private:
	__Board _board;
public:
	RandomPlayer() : BasePlayer() {
		_board = __Board();
	}

	~RandomPlayer() {

	}

	void usi() {
		std::cout << "info got cmd = usi" << std::endl;
		std::cout << "id name RandomPlayerV0.02.1" << std::endl;
		std::cout << "id author lefu777" << std::endl;
		std::cout << "usiok" << std::endl;
	}

	void setoption(std::string cmd) {
		std::cout << "info got cmd = setoption" << std::endl;
		std::cout << "info NotImplemented" << std::endl;
	}

	// TODO: �{���́A������init �Ƃ��ĂԁB
	bool _isready() {
		_board.reset();
		return true;
	}

	void isready() {
		if (_isready()) {
			std::cout << "readyok" << std::endl;
		}
		else {
			std::cout << "Error: isready failed" << std::endl;;
			exit(1);
		}
	}

	void _usinewgame() {
		
	}

	void usinewgame() {
		_usinewgame();
		std::cout << "info got cmd = usinewgame" << std::endl;
	}

	// @arg moves
    //     : �ȉ���2�ʂ肪����B
    //       startpos moves xxxx xxxx ...
    //       [sfen�`���̔C�ӂ̏����ǖ�(�擪��sfen�͖���)] moves xxxx xxxx ...
	virtual void position(const std::string& moves_str) {
		//std::cout << "info got cmd = position" << std::endl;
		if (!_board.set_position(moves_str)) {
			std::cout << "info Error: failed to set_position" << std::endl;
		}
	}

	void direct_push(u32 move) {
		_board.push(move);
	}


	static int _get_random_move(__Board& board) {
#ifdef DEBUG
		assert(!(board.is_game_over() || board.is_nyugyoku()));
#endif
		__LegalMoveList legal_moves(board);
		int legal_moves_size = legal_moves.size();
		if (legal_moves_size == 1) {
			return legal_moves.move();
		}
		else {
			int random_move_idx = mt_genrand_int32() % (legal_moves_size - 1);
			while (random_move_idx > 0) {
				legal_moves.next();
				--random_move_idx;
			}
			return legal_moves.move();
		}
	}

	// https://github.com/TadaoYamaoka/python-dlshogi/blob/master/pydlshogi/player/mcts_player.py
	// �I�� or �錾���� �̋ǖʂł͌Ăяo���Ȃ��B
	int _go() {
		return _get_random_move(_board);
	}

	void go() {
		if (_board.is_game_over()) {    // ���@�肪0��
			std::cout << "bestmove resign" << std::endl;
		}
		else if (_board.is_nyugyoku()) {
			std::cout << "bestmove win" << std::endl;
		}
		else {
			std::cout << "bestmove " << __move_to_usi(_go()) << std::endl;
		}
	}
};

