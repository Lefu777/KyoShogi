#pragma once
#include <iostream>
#include <iomanip>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef NOMINMAX
#include "cshogi.h"
#include "MT.hpp"


// �w���萶���Ղ�(move generation festival)
// �����ɍ��@�萶���̎��Ԃ��v������
// @arg sfen: �擪��sfen �������Asfen�`���̋ǖ�
// @return
//     : �G���[�������0, ���������1��Ԃ��B
inline int mgf_test0(
	const std::string sfen = "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1",
	unsigned long long n_loops = 1e7
) {
	std::cout << "info: mgf_test0()" << std::endl;
	__Board board;
	board.set(sfen);
	__LegalMoveList legal_moves(board);
	int legal_moves_size = legal_moves.size();
	std::cout << board.dump() << std::endl;
	std::cout << "legal_moves_size = " << legal_moves_size << std::endl;

	LARGE_INTEGER freq, start, end;
	if (!QueryPerformanceFrequency(&freq)) {    // �P�ʏK��
		return 0;
	}
	if (!QueryPerformanceCounter(&start)) {
		return 0;
	}
	for (unsigned long long i = 0; i < n_loops; ++i) {
		__LegalMoveList legal_moves(board);
	}
	if (!QueryPerformanceCounter(&end)) {
		return 0;
	}
	double duration_sec = ((double)(end.QuadPart - start.QuadPart) / freq.QuadPart);
	double duration_ms = 1000 * duration_sec;
	std::cout << "duration = " << duration_ms << " [ms]\n";
	std::cout << "generation per sec = " << n_loops / duration_sec << std::setprecision(10) << " [/sec]\n";
	return 1;
}

// �w���萶���Ղ�(move generation festival)
// ���@�萶�� + random_move �̎擾 �̎��Ԃ��v��
// @arg sfen: �擪��sfen �������Asfen�`���̋ǖ�
// @return
//     : �G���[�������0, ���������1��Ԃ��B
inline int mgf_test1(
	const std::string sfen = "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1",
	unsigned long long n_loops = 1e7
) {
	std::cout << "info: mgf_test1()" << std::endl;
	__Board board;
	board.set(sfen);
	__LegalMoveList legal_moves(board);
	int legal_moves_size = legal_moves.size();
	std::cout << board.dump() << std::endl;
	std::cout << "legal_moves_size = " << legal_moves_size << std::endl;

	LARGE_INTEGER freq, start, end;
	if (!QueryPerformanceFrequency(&freq)) {    // �P�ʏK��
		return 0;
	}
	if (!QueryPerformanceCounter(&start)) {
		return 0;
	}
	for (unsigned long long i = 0; i < n_loops; ++i) {
		__LegalMoveList legal_moves(board);
		int random_move_idx = mt_genrand_int32() % (legal_moves_size - 1);
		while (random_move_idx > 0) {
			legal_moves.next();
			--random_move_idx;
		}
		auto m = legal_moves.move();
	}
	if (!QueryPerformanceCounter(&end)) {
		return 0;
	}
	double duration_sec = ((double)(end.QuadPart - start.QuadPart) / freq.QuadPart);
	double duration_ms = 1000 * duration_sec;
	std::cout << "duration = " << duration_ms << " [ms]\n";
	std::cout << "generation per sec = " << n_loops / duration_sec << std::setprecision(10) << " [/sec]\n";
	return 1;
}