#include "usi_parser.hpp"

#ifdef USI_ENGINE
// HACK: なんというか、これじゃあbase_playerの意味が無い。汚い実装。
void usi_loop() {
#ifdef RANDOM_PLAYER
	BasePlayer* player = new RandomPlayer();
#elif defined UCT_RANDOM_PLAYER
	BasePlayer* player = new UctRandomPlayer();
#endif

#if defined(RANDOM_PLAYER) || defined(UCT_RANDOM_PLAYER)
	// TODO
	//     : 本当は、player の関数は別thread で実行しないと、
	//       思考中のstop とか受け取れない。
	// stop に対応
	while (true) {
		std::string cmd, token;
		getline(std::cin, cmd);
		std::istringstream iss(cmd);
		iss >> token;

		if (token == "mode") {
			// 現在定義されているプリプロセッサの内の一部を、定義されていれば表示
#ifdef USI_ENGINE
			std::cout << "mode = USI_ENGINE" << std::endl;
#endif
#ifdef DEBUG
			std::cout << "mode = DEBUG" << std::endl;
#endif
		}
		else if (token == "usi") {
			player->usi();
		}
		else if (token == "setoption") {
			player->setoption(cmd.substr(10));
		}
		else if (token == "isready") {
			player->isready();
		}
		else if (token == "usinewgame") {
			player->usinewgame();
		}
		else if (token == "position") {
			player->position(cmd.substr(9));
		}
		// TODO; 持ち時間制御
		else if (token == "go") {
			player->go();
		}
		else if (token == "gameover") {
			;
		}
		else if (token == "quit") {
			exit(1);
		}
		else {
			std::cout << "ParserError: parser got unexpected cmd == [" << cmd << "]" << std::endl;;
			//exit(1);
		}
	}
#elif defined(PUCT_PLAYER)
	auto* player = new PuctPlayer();
	player->usi_loop();
#elif defined(PARALLEL_PUCT_PLAYER)
	auto* player = new ParallelPuctPlayer();
	player->usi_loop();
#endif
}
#endif

