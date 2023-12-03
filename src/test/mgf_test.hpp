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


// 指し手生成祭り(move generation festival)
// 純粋に合法手生成の時間を計測する
// @arg sfen: 先頭にsfen が無い、sfen形式の局面
// @return
//     : エラーがあれば0, 成功すれば1を返す。
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
	if (!QueryPerformanceFrequency(&freq)) {    // 単位習得
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

// 指し手生成祭り(move generation festival)
// 合法手生成 + random_move の取得 の時間を計測
// @arg sfen: 先頭にsfen が無い、sfen形式の局面
// @return
//     : エラーがあれば0, 成功すれば1を返す。
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
	if (!QueryPerformanceFrequency(&freq)) {    // 単位習得
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