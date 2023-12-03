#include <iostream>
#include <ctime>
#include <cstdlib>
#include <sstream>

#include "cshogi.h"

#include "config.hpp"
#include "MT.hpp"

#include "debug_string_queue.hpp"

// HACK
//     : main 関数が、TESTのプリプロセッサによる条件分岐で溢れてるので、
//       test_main() みたいな関数を、test_main.hpp とかに定義してそこにまとめる。

// NOTE
//     : 関数内でinclude してはならない。こういうglobal scope でinclude しよう。
//    -> 一見ビルドは出来るんだけど、
//       VSの色付ける奴とか、インテリセンスとかが機能せんくなったりする。
// FIXME: 何故か、定義していないのに、ていぎしている扱いになってる。。。
#ifdef USI_ENGINE
#include "usi_parser.hpp"
#elif defined GEN_SFEN
#include "gensfen_parser.hpp"
#elif defined MOVE_GENERATION_FESTIVAL
#include "mgf_test.hpp"
#elif defined VECTOR_TEST
#include "read_teacher_test.hpp"
#elif defined READ_TEACHER_TEST
#include "read_teacher_test.hpp"
#include "dataloader.h"
#elif defined SQ_TEST
#include <stdlib.h>
#elif defined DATALOADER_TEST
#include "dataloader.h"
#elif defined NN_TENSORRT_TEST
#include "nn_tensorrt.hpp"
#include "nn_tensorrt_test.hpp"
#elif defined MIF_TEST
#include "make_input_feature_test.hpp"
#elif defined PARSE_CSA_TEST
#include "parse_CSA_test.hpp"
#include "csa_to_teacher.hpp"
#elif defined LOAD_DLSHOGI_TEACHER_TEST
#include "load_dlshogi_teacher_test.hpp"
#elif defined READ_YOBOOK_TEST
#include "read_yobook_test.hpp"
#elif defined MIF_V1_TEST
#include "make_input_feature_FEATURE_V1_test.hpp"
#elif defined MIF_V2_TEST
#include "make_input_feature_FEATURE_V2_test.hpp"
#elif defined DFPN_TEST
#include "dfpn_test.hpp"
#elif defined CSHOGI_BOARD_MATE_MOVE_TEST
#include "cshogi_Board_mateMove_test.hpp"
#endif



#ifdef BB_TEST1
bool _check_state_info(__Board& board, std::string locate_info) {
	StateInfo* si = board.pos.get_state();
	for (int i = 0; i < board.ply() - 1; ++i) {
		si = si->previous;
		if (!(si != nullptr)) {
			std::cout << "Error: at " << locate_info << " linked list of StateInfo is discontinuous." << std::endl;
			std::cout << "liked list can be traced until i = " << i << " (if i == 0, completely cannot trace.)" << std::endl;
			std::cout << "ply = " << board.ply() << std::endl;
			std::cout << board.dump() << std::endl;
			exit(1);
		}
	}
}
#endif

int main() {
	// init
	initTable();
	Position::initZobrist(); 
	HuffmanCodedPos_init();
	PackedSfen_init();
	Book_init();
	Effect8_init();

	debug_queue_str_init();

	srand(time(NULL));       

	// std::cout << "info main" << std::endl;     

#ifdef USI_ENGINE
    usi_loop();
#elif defined GEN_SFEN
	gensfen_loop();
#elif defined BB_TEST1

	std::vector<__Board> board_vec;
	std::cout << board_vec.capacity() << std::endl;
	for (int i = 0; i < 2; ++i) {
		std::cout << "dec start" << std::endl;
		__Board board;
		std::cout << "dec done" << std::endl;
		auto move1 = board.move_from_usi("2g2f");
		board.push(move1);
		auto move2 = board.move_from_usi("3c3d");
		board.push(move2);
		_check_state_info(board, "normal");

		__Board& board_ref = board;
		_check_state_info(board_ref, "board_ref");

		__Board board_cpy = board;
		_check_state_info(board_cpy, "board_cpy");

		std::cout << "emplace_back()" << std::endl;
		board_vec.emplace_back(board);
		std::cout << board_vec.capacity() << std::endl;
	}
	//__Board& board_vec_0_ref = board_vec[0];
	__Board board_vec_0 = board_vec[0];
	_check_state_info(board_vec[0], "board_vec[0]");
	_check_state_info(board_vec[1], "board_vec[1]");
	//_check_state_info(board_vec_0_ref, "board_vec_0_ref");
	_check_state_info(board_vec_0, "board_vec_0");

	board_vec.clear();
	std::cout << board_vec.capacity() << std::endl;

	std::cout << "test all done" << std::endl;
#elif defined BB_TEST2
	// board から(もっと言うとposition)、盤面の情報を取得する。
	__Board board;
	std::string cmd, token;
	while (true) {
		getline(std::cin, cmd);
		std::istringstream iss(cmd);
		iss >> token;

	    if (token == "position") {
			std::string tmp = cmd.substr(9);
			board.set_position(tmp);
			auto& pos = board.pos;

			// NOTE: 2種類の方法がある。
#if false
			for (Color c = Black; c < ColorNum; ++c) {
				std::cout << "====================================================================================================" << std::endl;
				std::cout << "[color == " << c << "]" << std::endl;
				// NOTE: 手番が白なら、盤面を反転して入力特徴量を作成。
				const Color c2 = pos.turn() == Black ? c : oppositeColor(c);

				// NOTE
				//     : posより、各コマの配置をbitboard に格納。
				//       先後の区別はしない。
				//       つまり、PieceTypeNum個のbitboardの配列となる。
				// 駒の配置
				Bitboard bb[PieceTypeNum];
				for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
					// NOTE: pos から、color==c の各駒のbitboard を取得
					bb[pt] = pos.bbOf(pt, c);
				}

				for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
					std::cout << "[bitboard of " << PieceTypeToStr(pt) << ", " << pt << "]" << std::endl;
				//for (Square sq = SQ11; sq < SquareNum; ++sq) {
					// NOTE
					//     : アクセスするインデックスを変更することで、実質的に回転している。
					// 白の場合、盤面を180度回転
					//const Square sq2 = pos.turn() == Black ? sq : SQ99 - sq;

					for (Square sq = SQ11; sq < SquareNum; ++sq) {
					//for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
					//	std::cout << "[bitboard of " << PieceTypeToStr(pt) << "]" << std::endl;
						// 駒の配置
						if (bb[pt].isSet(sq)) {
							//(*features1)[c2][pt - 1][sq2] = 1;
							std::cout << "1 ";
						}
						else {
							std::cout << "0 ";
						}
						if ((sq + 1) % 9 == 0) {
							std::cout << std::endl;
						}
					}
				}
				// hand
				const Hand hand = pos.hand(c);
				int p = 0;    // 現在処理している持ち駒に対するチャンネルの先頭のindex
				for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
					u32 num = hand.numOf(hp);
					std::cout << "[HandPieceNum of " << HandPieceToStr(hp) << "] = " << num << std::endl;

					//if (num >= MAX_PIECES_IN_HAND[hp]) {
					//	// 上限値でclip
					//	num = MAX_PIECES_IN_HAND[hp];
					//}
					//std::fill_n((*features2_hand)[c2][p], (int)SquareNum * num, 1);
					//p += MAX_PIECES_IN_HAND[hp];
				}
			}
#else
			std::cout << board.dump();
			constexpr unsigned short N_SETW = 14;
			auto occupied_bb = pos.occupiedBB();
			Square prev_sq = SQ11;
			FOREACH_BB(occupied_bb, auto sq, {
				const Piece pc = pos.piece(sq);
				const PieceType pt = pieceToPieceType(pc);
				Color c = pieceToColor(pc);

				//// print
				for (Square sq_i = prev_sq + 1; sq_i < sq; ++sq_i) {
					std::cout << std::left << std::setw(N_SETW) << "0 ";
					if ((sq_i + 1) % 9 == 0) {
						std::cout << std::endl;
					}
				}
				std::ostringstream oss;
				oss << "[" << c << "," << PieceTypeToStr(pt) << "] ";
				std::cout << std::left << std::setw(N_SETW) << oss.str();
				if ((sq + 1) % 9 == 0) {
					std::cout << std::endl;
				}

				prev_sq = sq;
			});

			for (Color c = Black; c < ColorNum; ++c) {
				std::cout << "[color == " << c << "]" << std::endl;
				const Hand hand = pos.hand(c);
				int p = 0;    // 現在処理している持ち駒に対するチャンネルの先頭のindex
				for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
					u32 num = hand.numOf(hp);
					std::cout << "[HandPieceNum of " << HandPieceToStr(hp) << "] = " << num << std::endl;

					//if (num >= MAX_PIECES_IN_HAND[hp]) {
					//	// 上限値でclip
					//	num = MAX_PIECES_IN_HAND[hp];
					//}
					//std::fill_n((*features2_hand)[c2][p], (int)SquareNum * num, 1);
					//p += MAX_PIECES_IN_HAND[hp];
				}
			}
#endif
		}
		else if (token == "quit") {
			exit(1);
		}
		else {
			std::cout << "ParserError: parser got unexpected cmd == [" << cmd << "]" << std::endl;;
			//exit(1);
		}
	}
#elif defined MOVE_GENERATION_FESTIVAL
    if (!mgf_test0("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1", 5e7)) {
		std::cout << "Error: failed to mgf_test()" << std::endl;
	}
	system("pause");
#elif defined VECTOR_TEST
	if (!vector_back_test(1e9)) {
		std::cout << "Error: [vector_back_test()]" << std::endl;
	}
	system("pause");
#elif defined READ_TEACHER_TEST
 //   std::string x = "0123456789";
	//std::cout << x.substr(x.size() - 3) << std::endl;    // 後ろから3目、から最後まで
	//std::cout << x.size() << std::endl;
    std::string tmp;
	std::cin >> tmp;
	//read_teacher_test0("data/sfen_20230818174022_1426129347.bin");
	//if (!read_teacher_test1(tmp)) {
	//	std::cout << "Error: [read_teacher_test1()]" << std::endl;
	//}
	__FastDataloader fdl(tmp, 0);

	// shuffle前のをprint / shuffle
	//fdl.read_files();
	//fdl.print_teachers();
	//fdl.shuffle_files("tmp_data", "tmp_shuffle", "sfen");

	// print
	fdl.read_files();
	fdl.print_teachers();


	system("pause");
#elif defined SQ_TEST
    std::cout << "info: SQ_TEST" << std::endl;
	auto print_divmod = [](Square sq) {
		const div_t d = div(sq, 9);
		std::cout << "[sq = " << sq << "][(x,y) = " << d.quot << ", " << d.rem << "]" << std::endl;
		};
	print_divmod(SQ11);
	print_divmod(SQ19);
	print_divmod(SQ29);
	print_divmod(SQ39);

#elif defined DATALOADER_TEST
    std::cout << "info: DATALOADER_TEST" << std::endl;
	__FastDataloader fdl("data", 10000, 10000000);
	fdl.read_files_sequential();
#elif defined NN_TENSORRT_TEST
    std::cout << "info: NN_TENSORRT_TEST" << std::endl;
	NNTensorRT_test0();
	system("pause");
#elif defined MIF_TEST
    std::cout << "info: MIF_TEST" << std::endl;
	mif_test0();
#elif defined PARSE_CSA_TEST
    std::cout << "info: PARSE_CSA_TEST" << std::endl;
	//parse_csa_test0();
	CsaToTeacher ctt(3900, 72, PlyAndScore{ 32, 700 }, 1800, PONANZA);
	std::string tmp;
	std::cin >> tmp;
	ctt.csa_to_teacher(tmp, tmp + "_teacher", 16, 8192);

    system("pause");
#elif defined LOAD_DLSHOGI_TEACHER_TEST
    std::cout << "info: LOAD_DLSHOGI_TEACHER_TEST" << std::endl;
	std::string tmp;
	std::cin >> tmp;
	load_hcpex_test0(tmp);

	system("pause");
#elif defined READ_YOBOOK_TEST
    std::cout << "info: READ_YOBOOK_TEST" << std::endl;
    std::string tmp;
    std::cin >> tmp;
	read_yo_book_test1(tmp);

    system("pause");
#elif defined MIF_V1_TEST
    std::cout << "info: MIF_V1_TEST" << std::endl;
	make_input_features_FATURE_V1_test();
	system("pause");
#elif defined MIF_V2_TEST
    std::cout << "info: MIF_V2_TEST" << std::endl;
	//make_input_features_FATURE_V2_test();
	//make_input_features_FATURE_V2_test2();
	make_input_features_FATURE_V2_test3();
    system("pause");
#elif defined DFPN_TEST
	std::cout << "info: DFPN_TEST" << std::endl;
	//dfpn_test1();
	dfpn_bench0();
	//system("pause");
#elif defined CSHOGI_BOARD_MATE_MOVE_TEST
    std::cout << "info: CSHOGI_BOARD_MATE_MOVE_TEST" << std::endl;
    //dfpn_test1();
	cshogi_board_mate_move_test0();
    //system("pause");
#elif defined BOARD_TEST
    __Board board;
	board.set_position("startpos");
	std::cout << board.ply() << std::endl;

	const Move move = board.Move_from_usi("2g2f");
	board.push(move);
	std::cout << board.ply() << std::endl;

	board.set("lnS6/r4k1+B1/4pg1+L1/p2ps1P2/1p+b2ps2/P3P4/1PGPsP+n2/3G5/LNKL3+r1 b GPn6p 93");
	std::cout << board.dump() << std::endl;
	MoveList<Legal> ml(board.pos);
	std::cout << "final list" << std::endl;
	for (int i = 0; !ml.end(); ++ml, ++i) {
		std::cout << "[" << i << "] " << ml.move().toUSI() << std::endl;
	}

#elif defined MOVE_LABEL_TEST

#endif
}