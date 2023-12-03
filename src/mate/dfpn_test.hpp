#pragma once

#include <vector>
#include <thread>
#include <tuple>
#include <chrono>
#include <iostream>
#include <algorithm>

#include "cshogi.h"

#if defined DFPN_PARALLEL4 || defined DFPN_PARALLEL5
#include "dfpn_parallel_pv.hpp"
#endif

#include "stop_watch.hpp"
#include "util.hpp"

#include "debug_string_queue.hpp"



typedef std::tuple<std::string, std::string> sfenAndBestmove;
typedef std::tuple<std::string, std::string, int> sfenBestmoveAndDepth;

// �ЂƂ܂����Ⴀ�A�l�ݔ����܂ł̑��x���e�X�g����֐�����낤
// ���̏ꍇ�̓e�X�g�Ƃ������Abench ���ȁB

// �e�X�g�f�[�^�ɑ΂��ċl�ݒT�����s��
// @var test_data_vec
//     : �e�X�g�f�[�^�̃��X�g
//     : tuple([sfen], [���J����Ă��鐦���l���������v���O�����ɂ�鏉��̍őP��])
//     : TODO
//         : ���肪��ʂ肠��ꍇ��z�肵�Ă��Ȃ��B

#ifdef DFPN_PARALLEL3
inline void dfpn_test0(const int n_trials = 1) {
	std::vector<sfenAndBestmove> test_data_vec = {
		{"8+n/4+N3l/9/9/9/9/p7k/1p2L4/K1+p1+NL+NL1 b 2r2b4g4s15p 1"             , "3i2h"},
		{"lnsgk2nl/7+R1/p1pppp2p/6p2/9/2P5P/P2PPPP2/9/+r1SGKGSNL b GS2P2bnl2p 1", "G*4a"},
		{"9/8R/7k1/8R/9/1l7/np7/K1p6/9 b 2b4g4s3n3l16p 1"                       , "1b1c+"},
		{"5+r2k/9/9/9/9/2+b5+b/p8/1p7/K1+p4+R1 b S8P4g3s4n4l7p 1"               , "P*1b"},
		{"8k/9/9/9/5Llll/9/p8/1p7/K1+p6 b 2G2S2N3P2r2b2g2s2n12p 1"              , "P*1b"},
		{"9/9/4+B4/9/5Rp1k/9/4+R4/Pp7/KP7 b Pb4g4s4n4l13p 1"                    , "4e3e"},
		{"9/9/+R8/4k4/9/+R8/8p/7p1/6+p1K b Bb4g4s4n4l15p 1"                     , "B*7b"},     // (depth=41)
		{"9/1l6n/9/2K2Sln1/8b/6+l2/2G4k1/9/Gr2+b+r2S b 2G2S2NL18P 1"            , "G*2h"},     // (depth = 49)
		{"9/9/3R5/5k3/2B6/3r1B3/9/Pp7/KP7 b 4g4s4n4l15p 1"                      , "7e6f"},     // (depth=47)
		{"8k/r8/9/9/6R2/9/9/Pp7/K+P7 b 4S3N2P2b4gn4l13p 1"                      , "3e3a+"},    // (depth=51)
	};

	__Board board;
	__DfPn dfpn;
	stopwatch_t sw;
	bool is_successful = true;
	double time_total = 0;
	uint64_t nodes_total = 0;
	std::vector<double> time_vec;
	std::vector<uint64_t> nodes_vec;

	dfpn.set_max_search_node(3200'0000);
	dfpn.set_maxdepth(51);
	dfpn.set_draw_ply(512);
	dfpn.set_hashsize(65536);
	//dfpn.set_hashsize(8192);

	std::cout << "info: " << dfpn.get_option_str() << std::endl;
	std::cout << "info: n_trials=" << n_trials << std::endl;
	std::cout << "info: loop start." << std::endl;
	const int data_size = test_data_vec.size();
	for (int i = 0; i < data_size; ++i) {
		const auto& test_data = test_data_vec[i];
		const auto& sfen = std::get<0>(test_data);
		const auto& expected_bestmove_usi = std::get<1>(test_data);

		board.set(sfen);

		std::cout << "[" << i + 1 << "/" << data_size << "] " << sfen << std::endl;

		std::string pv_str;
		int depth;
		Move bestmove;
		bool is_mate = false;
		double time = 0;
		int64_t nodes = 0;
#ifdef DFPN_PARALLEL3
		int64_t searcher_node;
#endif
		for (int j = 0; j < n_trials; ++j) {
			sw.start();
#if defined(DFPN_CSHOGI) || defined(DFPN_DLSHOGI) || defined(DFPN_PARALLEL)
#else
			dfpn.new_search();
#endif
			if (j == 0) {
				// ����̌��ʂ��̗p
#ifdef DFPN_PARALLEL3
				is_mate = dfpn.search(board, searcher_node, -1);
#else
				is_mate = dfpn.search(board);
#endif
				sw.stop();    // get_pv �͏d���̂ŁA������O�Ŏ~�߂�
				std::tie(pv_str, depth, bestmove) = dfpn.get_pv(board);
			}
			else {
#ifdef DFPN_PARALLEL3
				int64_t tmp;
				is_mate = dfpn.search(board, tmp, -1);
#else
				is_mate = dfpn.search(board);
#endif
				sw.stop();
			}

#if defined(DFPN_CSHOGI) || defined(DFPN_DLSHOGI)
#else
			if (n_trials > 1) {
				dfpn.reset();
			}
#endif

			//std::cout
			//	<< "       "
			//	<< " time " << std::right << std::setw(6) << positive_decimal_to_int_str(sw.elapsed_ms())
			//	<< " nodes " << std::right << std::setw(8) << dfpn.get_searched_node()
			//	<< " depth " << std::right << std::setw(3) << depth
			//	<< " now " << get_now_str()
			//	<< std::endl;

			time += sw.elapsed_ms();
#ifdef DFPN_PARALLEL3
			nodes += searcher_node;
#else
			nodes += dfpn.get_searched_node();
#endif
		}

		const auto nps = 1000 * (nodes / time);
		time /= n_trials;     // ���ς����
		nodes /= n_trials;    // ���ς����
		time_vec.emplace_back(time);
		nodes_vec.emplace_back(nodes);

		// bestmove = Move(0b101'0000'0001);    // debug

		time_total += time;
		nodes_total += nodes;

		std::cout
			<< "       "
			<< " time " << std::right << std::setw(6) << positive_decimal_to_int_str(time)
			<< " nodes " << std::right << std::setw(8) << nodes
			<< " nps " << std::right << std::setw(6) << positive_decimal_to_int_str(nps)
			<< " depth " << std::right << std::setw(3) << depth
			<< " now " << get_now_str()
			<< std::endl;
		if (!is_mate) {
			std::cout << "    Error: expected is_mate must be true, but got false." << std::endl;
			is_successful = false;
		}
		if (__move_to_usi(bestmove.value()) != expected_bestmove_usi) {
			std::cout
				<< "    Warninig: expected bestmove is " << expected_bestmove_usi
				<< ", but got " << __move_to_usi(bestmove.value())
				<< ". pv = " << pv_str
				<< std::endl;
			is_successful = false;
		}
	}
	std::cout << "info: loop done." << std::endl;

	std::cout << "info: total time = " << positive_decimal_to_int_str(time_total) << std::endl;
	std::cout << "info: total nodes = " << positive_decimal_to_int_str(nodes_total) << std::endl;
	std::cout << "info: average time = " << positive_decimal_to_int_str(time_total / data_size) << std::endl;
	std::cout << "info: average nodes = " << positive_decimal_to_int_str(nodes_total / data_size) << std::endl;
	std::cout << "info: nps = " << positive_decimal_to_int_str(1000 * (nodes_total / time_total)) << std::endl;

	for (int i = 0; i < data_size; ++i) {
		const auto& time_tmp = std::accumulate(time_vec.begin(), time_vec.begin() + i + 1, 0.);
		const auto& nodes_tmp = std::accumulate(nodes_vec.begin(), nodes_vec.begin() + i + 1, 0);
		std::cout
			<< "[" << i + 1 << "/" << data_size << "]"
			<< " time " << std::right << std::setw(6) << positive_decimal_to_int_str(time_tmp)
			<< " nodes " << std::right << std::setw(9) << nodes_tmp
			<< " average time " << std::right << std::setw(6) << positive_decimal_to_int_str(time_tmp / (i + 1))
			<< " average nodes " << std::right << std::setw(8) << nodes_tmp / (i + 1)
			<< " nps " << std::right << std::setw(6) << positive_decimal_to_int_str(1000 * (nodes_tmp / time_tmp))
			<< std::endl;
	}

	if (is_successful) {
		std::cout << "info: The test was successful." << std::endl;
	}
	else {
		std::cout << "Error: Test failed." << std::endl;
	}
}
#endif    // DFPN_PARALLEL3


const std::string NO_MATE = "none";
inline void dfpn_test1() {
	// ��ɍs�������s���Ԃ��Z��
	// TODO
	//     : 9�Ԗڂ̋ǖ�(depth=47) �́ATHPN_MULTIPLY ��2 ���ƃ}���`�X���b�h�̌��ʂ������A3 ����32 -> 9 �b���炢�ɂȂ�B
	//       10�Ԗڂ̋ǖ�(depth=51) �́ATHPN_MULTIPLY ��1 ���ƃ}���`�X���b�h�̌��ʂ������A2, 3 ����49 -> 6 �b���炢�ɂȂ�B
	//       -> ��1. num_searched �������ǖʂł͑��₷...�H
	//          ��2. ���萔�������ǖʂł͑��₷...?
	//     : �܂��ł��A���������œ����������ɗ�������A�Ԉ�����肷��̂��ǂ��ɂ����Ȃ��ƁB
	//       �ŏ���6�̓�����ŗ�����z������΁A�����Ńf�o�b�O�������B
	//       -> �܂��A���񎞂Ƀo�O��ǖʂ�T���B
	//          �ԈႤ�m���������ǖʂ��s�b�N�A�b�v���ׂ������H
	//          ��������A���enqueue_debug_str() �Ȃ�Ȃ�Ȃ���g���āAtarget�ǖʂ�pn = inf �Ƃ���Ă�ӏ���T���B
	//     : ��6 ��200�񂮂炢���点�����ǃo�O��Ȃ������B
	//       7�Ԗڂ́A30~60�񂮂炢�͑��点���Ǝv�����ǁA�o�O��Ȃ������B
	//       8�Ԗڂ̓o�O�����B����ŁB
	std::vector<sfenBestmoveAndDepth> test_data_vec = {
		//{"+B2B1n2K/7+R1/p2p1p1ps/3g2+r1k/1p3n3/4n1P+s1/PP7/1S6p/L7L b 3GS7Pn2l2p 1", NO_MATE, -1},
		{"8+n/4+N3l/9/9/9/9/p7k/1p2L4/K1+p1+NL+NL1 b 2r2b4g4s15p 1"                , "3i2h" , 37},
		{"lnsgk2nl/7+R1/p1pppp2p/6p2/9/2P5P/P2PPPP2/9/+r1SGKGSNL b GS2P2bnl2p 1"   , "G*4a" , 9},
		{"9/8R/7k1/8R/9/1l7/np7/K1p6/9 b 2b4g4s3n3l16p 1"                          , "1b1c+", 27},
		{"5+r2k/9/9/9/9/2+b5+b/p8/1p7/K1+p4+R1 b S8P4g3s4n4l7p 1"                  , "P*1b" , 41},
		{"8k/9/9/9/5Llll/9/p8/1p7/K1+p6 b 2G2S2N3P2r2b2g2s2n12p 1"                 , "P*1b" , 31},
		{"9/9/4+B4/9/5Rp1k/9/4+R4/Pp7/KP7 b Pb4g4s4n4l13p 1"                       , "4e3e" , 31},
		{"9/9/+R8/4k4/9/+R8/8p/7p1/6+p1K b Bb4g4s4n4l15p 1"                        , "B*7b" , 41},  
		{"l7l/7+B1/2nkg3p/pr3s1P1/2P1n1P1L/P1Spbg3/+p2P3R1/3KSG3/LN7 w GSn9p 1"    , "6f6g+", 39},
		{"9/1l6n/9/2K2Sln1/8b/6+l2/2G4k1/9/Gr2+b+r2S b 2G2S2NL18P 1"               , "G*2h" , 49},
		{"9/9/3R5/5k3/2B6/3r1B3/9/Pp7/KP7 b 4g4s4n4l15p 1"                         , "7e6f" , 47}, // get_pv �ŗ������B(-> stack overflow)
		{"8k/r8/9/9/6R2/9/9/Pp7/K+P7 b 4S3N2P2b4gn4l13p 1"                         , "3e3a+", 51},
	};
	// get_pv �����s���āA�l�݃G���W���̐��������`�F�b�N����B
	bool enable_get_pv = false;

	__Board board;
	ParallelDfpnGroup dfpn_group;
	stopwatch_t sw;
	bool is_successful = true;
	double time_total = 0;
	uint64_t nodes_total = 0;
	std::vector<double> time_vec;
	std::vector<uint64_t> nodes_vec;
	
	std::cout << "info: sizeof(ns_dfpn::TTEntry) = " << sizeof(ns_dfpn::TTEntry) << std::endl;

	ns_dfpn::TranspositionTable _TT;
	_TT.Resize(
		32768
	);
	_TT.Clear();

	int n_threads = 24;
	dfpn_group.init(
		n_threads,
		51,
		50000000,
		512
	);
	dfpn_group.set_tt(&_TT);
	
	//dfpn.set_hashsize(8192);

	std::cout << "info: n_threads = " << n_threads << std::endl;
	std::cout << "info: loop start." << std::endl;
	//std::this_thread::sleep_for(std::chrono::milliseconds(1024));

	const int data_size = test_data_vec.size();
	for (int i = 0; i < data_size; ++i) {
		const auto& test_data = test_data_vec[i];
		const auto& sfen = std::get<0>(test_data);
		const auto& expected_bestmove_usi = std::get<1>(test_data);
		const auto& expected_min_depth = std::get<2>(test_data);

		board.set(sfen);

		std::cout << "[" << i + 1 << "/" << data_size << "] " << sfen << std::endl;

		std::string pv_str;
		int depth = 0;
		Move bestmove(0);
		bool is_mate = false;

		dfpn_group.set_root_board(board);

		sw.start();
		std::cout
			<< "       "
			<< " start " << get_now_str() << std::endl;
		//std::cout << "info: gen = " << dfpn_group.get_generation() << std::endl;

		_TT.NewSearch();
		dfpn_group.run();
		dfpn_group.join();
		is_mate = dfpn_group.get_is_mate();

		sw.stop();    // get_pv �͏d���̂ŁA������O�Ŏ~�߂�
		if (enable_get_pv) {
			//// debug
			//std::cout << "        get_pv() start." << std::endl;

			std::tie(pv_str, depth, bestmove) = dfpn_group.get_pv(board);

			//// debug
			//std::cout << "        get_pv() done." << std::endl;
		}

		double time = sw.elapsed_ms();
		int64_t nodes = dfpn_group.get_searched_node();
		const auto nps = 1000 * (nodes / time);
		time_vec.emplace_back(time);
		nodes_vec.emplace_back(nodes);

		time_total += time;
		nodes_total += nodes;

		std::cout
			<< "       "
			<< " time " << std::right << std::setw(6) << positive_decimal_to_int_str(time)
			<< " nodes " << std::right << std::setw(8) << nodes
			<< " nps " << std::right << std::setw(6) << positive_decimal_to_int_str(nps)
			<< " depth " << std::right << std::setw(3) << depth
			//<< " now " << get_now_str()
			<< std::endl;
		std::cout
			<< "       "
			<< " done  " << get_now_str() << std::endl;
		//std::cout
		//	<< "       "
		//	<< " pv = [" << pv_str << "]"
		//	<< std::endl;
		if ((!is_mate && expected_bestmove_usi != NO_MATE) || ((is_mate && expected_bestmove_usi == NO_MATE))) {
			std::cout << "    Error: result is unexpected is_mate." << std::endl;
			is_successful = false;
		}
		if (enable_get_pv) {
			// �l�݂̎��̂݁A�l�ݎ菇�ɂ��Ă��`�F�b�N����B
			if (expected_bestmove_usi != NO_MATE && __move_to_usi(bestmove.value()) != expected_bestmove_usi) {
				std::cout
					<< "    Warninig: expected bestmove is " << expected_bestmove_usi
					<< ", but got " << __move_to_usi(bestmove.value())
					<< ". pv = " << pv_str
					<< std::endl;
				is_successful = false;
				flush_debug_str();
			}
			else if (expected_bestmove_usi != NO_MATE && depth < expected_min_depth) {
				std::cout
					<< "    Warninig: expected min depth is " << expected_min_depth
					<< ", but got " << depth
					<< ". pv = " << pv_str
					<< std::endl;
				is_successful = false;
				flush_debug_str();
			}
			else {
				//// debug
				//std::cout << "    info: pv = " << pv_str << std::endl;
			}
		}
	}
	std::cout << "info: loop done." << std::endl;

	std::cout << "info: total time = " << positive_decimal_to_int_str(time_total) << std::endl;
	std::cout << "info: total nodes = " << positive_decimal_to_int_str(nodes_total) << std::endl;
	std::cout << "info: average time = " << positive_decimal_to_int_str(time_total / data_size) << std::endl;
	std::cout << "info: average nodes = " << positive_decimal_to_int_str(nodes_total / data_size) << std::endl;
	std::cout << "info: nps = " << positive_decimal_to_int_str(1000 * (nodes_total / time_total)) << std::endl;

	for (int i = 0; i < data_size; ++i) {
		const auto& time_tmp = std::accumulate(time_vec.begin(), time_vec.begin() + i + 1, 0.);
		const auto& nodes_tmp = std::accumulate(nodes_vec.begin(), nodes_vec.begin() + i + 1, 0);
		std::cout
			<< "[" << i + 1 << "/" << data_size << "]"
			<< " time " << std::right << std::setw(6) << positive_decimal_to_int_str(time_tmp)
			<< " nodes " << std::right << std::setw(9) << nodes_tmp
			<< " average time " << std::right << std::setw(6) << positive_decimal_to_int_str(time_tmp / (i + 1))
			<< " average nodes " << std::right << std::setw(8) << nodes_tmp / (i + 1)
			<< " nps " << std::right << std::setw(6) << positive_decimal_to_int_str(1000 * (nodes_tmp / time_tmp))
			<< std::endl;
	}

	if (is_successful) {
		std::cout << "info: The test was successful." << std::endl;
	}
	else {
		std::cout << "Error: Test failed!!!!!" << std::endl;
	}
}

inline void dfpn_bench0() {
	// ��ɍs�������s���Ԃ��Z��
	// TODO
	//     : 9�Ԗڂ̋ǖ�(depth=47) �́ATHPN_MULTIPLY ��2 ���ƃ}���`�X���b�h�̌��ʂ������A3 ����32 -> 9 �b���炢�ɂȂ�B
	//       10�Ԗڂ̋ǖ�(depth=51) �́ATHPN_MULTIPLY ��1 ���ƃ}���`�X���b�h�̌��ʂ������A2, 3 ����49 -> 6 �b���炢�ɂȂ�B
	//       -> ��1. num_searched �������ǖʂł͑��₷...�H
	//          ��2. ���萔�������ǖʂł͑��₷...?
	//     : �܂��ł��A���������œ����������ɗ�������A�Ԉ�����肷��̂��ǂ��ɂ����Ȃ��ƁB
	//       �ŏ���6�̓�����ŗ�����z������΁A�����Ńf�o�b�O�������B
	//       -> �܂��A���񎞂Ƀo�O��ǖʂ�T���B
	//          �ԈႤ�m���������ǖʂ��s�b�N�A�b�v���ׂ������H
	//          ��������A���enqueue_debug_str() �Ȃ�Ȃ�Ȃ���g���āAtarget�ǖʂ�pn = inf �Ƃ���Ă�ӏ���T���B
	//     : ��6 ��200�񂮂炢���点�����ǃo�O��Ȃ������B
	//       7�Ԗڂ́A30~60�񂮂炢�͑��点���Ǝv�����ǁA�o�O��Ȃ������B
	//       8�Ԗڂ̓o�O�����B����ŁB
	//     :
	// NOTE 
	//     : NO_MATE �͏����āA�ォ��4��, 5�� �͓��ɁARepSup �Ƃ̍����̊֌W�ŕs�l�݂Ɣ��肳��邱�Ƃ�����(50 ~ 100 ���1��)�B
	//       is_mate == false �Ȃ玎�s�񐔂�+1 ���āA���̉�̎��s�͖����������ƂɁB
	std::vector<sfenBestmoveAndDepth> test_data_vec = {
		//{"+B2B1n2K/7+R1/p2p1p1ps/3g2+r1k/1p3n3/4n1P+s1/PP7/1S6p/L7L b 3GS7Pn2l2p 1"      , NO_MATE, -1},
		{"8+n/4+N3l/9/9/9/9/p7k/1p2L4/K1+p1+NL+NL1 b 2r2b4g4s15p 1"                      , "3i2h" , 37},
		{"lnsgk2nl/7+R1/p1pppp2p/6p2/9/2P5P/P2PPPP2/9/+r1SGKGSNL b GS2P2bnl2p 1"         , "G*4a" , 9},
		{"lr7/3g1+N+B2/p3p4/1pp4kp/P2S1p3/2P1N1s1P/1PGP1P3/1+bN1KGpp1/+r7L w 2Pg2snl2p 132", "G*5i" , 9},      // �o�O���Ă�...? -> 1����񂵂�ok�������̂�PvDfpn���ȁB
		{"lr7/3g1+N3/p3p+B3/1pp4kp/P2S1p3/2P1N1s1P/1PGP1P3/1+bN1KGpp1/+r7L w Pg2snl3p 132", "G*5i" , 9},
		{"9/8R/7k1/8R/9/1l7/np7/K1p6/9 b 2b4g4s3n3l16p 1"                                , "1b1c+", 27},
		{"5+r2k/9/9/9/9/2+b5+b/p8/1p7/K1+p4+R1 b S8P4g3s4n4l7p 1"                        , "P*1b" , 41},    // �o�O���Ă�B
		{"8k/9/9/9/5Llll/9/p8/1p7/K1+p6 b 2G2S2N3P2r2b2g2s2n12p 1"                       , "P*1b" , 31},    // �o�O���Ă�B
		{"9/9/4+B4/9/5Rp1k/9/4+R4/Pp7/KP7 b Pb4g4s4n4l13p 1"                             , "4e3e" , 31},
		{"9/9/+R8/4k4/9/+R8/8p/7p1/6+p1K b Bb4g4s4n4l15p 1"                              , "B*7b" , 41},
		{"l7l/7+B1/2nkg3p/pr3s1P1/2P1n1P1L/P1Spbg3/+p2P3R1/3KSG3/LN7 w GSn9p 1"          , "6f6g+", 39},
		{"9/1l6n/9/2K2Sln1/8b/6+l2/2G4k1/9/Gr2+b+r2S b 2G2S2NL18P 1"                     , "G*2h" , 49},
		{"9/9/3R5/5k3/2B6/3r1B3/9/Pp7/KP7 b 4g4s4n4l15p 1"                               , "7e6f" , 47}, // get_pv �ŗ������B(-> stack overflow)
		{"8k/r8/9/9/6R2/9/9/Pp7/K+P7 b 4S3N2P2b4gn4l13p 1"                               , "3e3a+", 51},
	};
	// get_pv �����s���āA�l�݃G���W���̐��������`�F�b�N����B
	bool enable_get_pv = false;

	// ���s��
	int n_trials = 1;

	__Board board;
	ParallelDfpnGroup dfpn_group;
	stopwatch_t sw;
	bool is_successful = true;
	double time_total = 0;
	uint64_t nodes_total = 0;
	std::vector<double> time_vec;
	std::vector<uint64_t> nodes_vec;

	std::cout << "info: sizeof(ns_dfpn::TTEntry) = " << sizeof(ns_dfpn::TTEntry) << std::endl;

	ns_dfpn::TranspositionTable _TT;
	_TT.Resize(
		8192
	);
	_TT.Clear();

	// �X���b�h��
	int n_threads = 1;
	dfpn_group.init(
		n_threads,
		51,
		50000000,
		512
	);
	dfpn_group.set_tt(&_TT);

	//dfpn.set_hashsize(8192);

	std::cout << "info: n_threads = " << n_threads << std::endl;
	std::cout << "info: n_trials = " << n_trials << std::endl;
	std::cout << "info: loop start." << std::endl;
	//std::this_thread::sleep_for(std::chrono::milliseconds(1024));

	const int data_size = test_data_vec.size();
	for (int i = 0; i < data_size; ++i) {
		const auto& test_data = test_data_vec[i];
		const auto& sfen = std::get<0>(test_data);
		const auto& expected_bestmove_usi = std::get<1>(test_data);
		const auto& expected_min_depth = std::get<2>(test_data);

		board.set(sfen);

		std::cout << "[" << i + 1 << "/" << data_size << "] " << sfen << std::endl;

		std::string pv_str;
		int depth = 0;
		Move bestmove(0);


		double time = 0;
		int64_t nodes = 0;
		double nps = 0;

		dfpn_group.set_root_board(board);

		std::cout
			<< "       "
			<< " start " << get_now_str() << std::endl;

		auto lim = n_trials;
		for (int j = 0; j < lim; ++j) {
			std::cout << "\rtrial run  [" << j + 1 << "/" << lim << "]" << std::flush;
			sw.start();
			_TT.NewSearch();
			dfpn_group.run();
			dfpn_group.join();
			bool is_mate = dfpn_group.get_is_mate();
			sw.stop();
			//std::cout << "\rtrial done [" << j + 1 << "/" << n_trials << "]" << std::flush;

			time += sw.elapsed_ms();
			nodes += dfpn_group.get_searched_node();
			nps += 1000 * (dfpn_group.get_searched_node() / sw.elapsed_ms());
			//std::cout << "time " << sw.elapsed_ms() << std::endl;

			// �l�݂Ȃ̂ɕs�l�݂�Ԃ��� || �s�l�݂Ȃ̂ɋl�݂�Ԃ���
			if ((!is_mate && expected_bestmove_usi != NO_MATE) || ((is_mate && expected_bestmove_usi == NO_MATE))) {
				std::cout << "\nError!!: result is unexpected is_mate." << std::endl;
				is_successful = false;

				// ����͖����������ƂɁB
				time -= sw.elapsed_ms();
				nodes -= dfpn_group.get_searched_node();
				nps -= 1000 * (dfpn_group.get_searched_node() / sw.elapsed_ms());
				++lim;
			}

		}
		std::cout << "\r" << std::flush;

		time_vec.emplace_back(time);
		nodes_vec.emplace_back(nodes);

		time_total += time;
		nodes_total += nodes;

		//// 3�Ƃ��d�v
		std::cout
			<< "       "
			<< " time_per_trial " << std::right << std::setw(6) << positive_decimal_to_int_str(time / n_trials)
			<< " nodes_per_trial " << std::right << std::setw(8) << nodes / n_trials
			<< " nps_per_trial " << std::right << std::setw(7) << positive_decimal_to_int_str(nps / n_trials)
			//<< " now " << get_now_str()
			<< std::endl;
		std::cout
			<< "       "
			<< " done  " << get_now_str() << std::endl;
		//std::cout
		//	<< "       "
		//	<< " pv = [" << pv_str << "]"
		//	<< std::endl;
	}
	std::cout << "info: loop done." << std::endl;

	//// ��2���d�v
	std::cout << "info: total time = " << positive_decimal_to_int_str(time_total) << std::endl;
	std::cout << "info: total nodes = " << positive_decimal_to_int_str(nodes_total) << std::endl;
	std::cout << "info: time per trial = " << positive_decimal_to_int_str(time_total / n_trials) << std::endl;
	std::cout << "info: nodes per trial = " << positive_decimal_to_int_str(nodes_total / n_trials) << std::endl;
	std::cout << "info: time per trial per data = " << positive_decimal_to_int_str((time_total / n_trials) / data_size) << std::endl;
	std::cout << "info: nodes per trial per data = " << positive_decimal_to_int_str((nodes_total / n_trials) / data_size) << std::endl;
	std::cout << "info: nps = " << positive_decimal_to_int_str(1000 * (nodes_total / time_total)) << std::endl;

	for (int i = 0; i < data_size; ++i) {
		const auto& time_tmp = std::accumulate(time_vec.begin(), time_vec.begin() + i + 1, 0.);
		const auto& nodes_tmp = std::accumulate(nodes_vec.begin(), nodes_vec.begin() + i + 1, 0);
		std::cout
			<< "[" << i + 1 << "/" << data_size << "]"
			<< " total time " << std::right << std::setw(6) << positive_decimal_to_int_str(time_tmp)
			<< " total nodes" << std::right << std::setw(12) << nodes_tmp
			<< " time per trial" << std::right << std::setw(6) << positive_decimal_to_int_str(time_tmp / n_trials)
			<< " nodes per trial" << std::right << std::setw(11) << nodes_tmp / n_trials
			<< " time per trial per data" << std::right << std::setw(6) << positive_decimal_to_int_str((time_tmp / n_trials) / (i + 1))
			<< " nodes per trial per data" << std::right << std::setw(8) << (nodes_tmp / n_trials) / (i + 1)
			<< " nps " << std::right << std::setw(6) << positive_decimal_to_int_str(1000 * (nodes_tmp / time_tmp))
			<< std::endl;
	}

	if (is_successful) {
		std::cout << "info: The test was successful." << std::endl;
	}
	else {
		std::cout << "Error: Test failed!!!!!" << std::endl;
	}
}

